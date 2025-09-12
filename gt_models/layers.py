import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.utils import to_dense_batch


class GPSLayer(nn.Module):
    """GPS Layer: 结合局部MPNN和全局Transformer的图层
    
    基于PyG官方教程的GPS架构实现
    """
    def __init__(self, hidden_dim: int, heads: int = 8, dropout: float = 0.1, 
                 mpnn_type: str = 'gat'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        
        # 局部MPNN层
        if mpnn_type == 'gat':
            self.local_conv = GATConv(hidden_dim, hidden_dim // heads, heads=heads, 
                                    concat=True, dropout=dropout, add_self_loops=False)
        elif mpnn_type == 'gcn':
            self.local_conv = GCNConv(hidden_dim, hidden_dim)
        elif mpnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim)
            )
            self.local_conv = GINEConv(mlp, edge_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported MPNN type: {mpnn_type}")
            
        # 全局Transformer层
        self.global_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """GPS层前向传播
        
        Args:
            x: 节点特征 [N, hidden_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边属性/边类型编码 [E, hidden_dim] (可选)
            batch: 批次信息 [N] (可选)
        """
        hs = []
        
        # 1. 局部MPNN
        if isinstance(self.local_conv, GINEConv) and edge_attr is not None:
            # GINEConv支持边属性
            h = self.local_conv(x, edge_index, edge_attr)
        else:
            # GAT和GCN不直接支持边属性，但可以通过注意力机制间接利用
            h = self.local_conv(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # 残差连接
        h = self.norm1(h)
        hs.append(h)
        
        # 2. 全局Transformer
        h_local = hs[0]  # 使用局部MPNN的输出
        if batch is not None:
            # 处理批次数据
            h_dense, mask = to_dense_batch(h_local, batch)
            h_attn, _ = self.global_attn(h_dense, h_dense, h_dense, 
                                       key_padding_mask=~mask, need_weights=False)
            h = h_attn[mask]
        else:
            # 单图处理
            h_dense = h_local.unsqueeze(0)
            h_attn, _ = self.global_attn(h_dense, h_dense, h_dense, need_weights=False)
            h = h_attn.squeeze(0)
            
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + h_local  # 残差连接使用局部输出
        h = self.norm2(h)
        hs.append(h)
        
        # 3. 合并局部和全局输出
        out = sum(hs)
        
        # 4. MLP层
        out = out + self.mlp(self.norm3(out))
        
        return out