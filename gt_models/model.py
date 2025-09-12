import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .layers import GPSLayer

class GraphTransformer(nn.Module):
    """基于GPS的图Transformer模型
    
    使用PyG官方推荐的GPS架构，结合局部MPNN和全局Transformer
    """
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int = 4, 
                 heads: int = 8, dropout: float = 0.1, mpnn_type: str = 'gat'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点嵌入
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # 边类型嵌入 - 支持正负边编码
        self.edge_type_embedding = nn.Embedding(3, hidden_dim)  # 3种边类型：正(1)、负(-1)、null(0)
        
        # GPS层
        self.gps_layers = nn.ModuleList([
            GPSLayer(hidden_dim, heads, dropout, mpnn_type) 
            for _ in range(num_layers)
        ])
        
        # 边分类器 - 3类分类
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3个类别：正、负、null
        )
        
    def forward(self, x, edge_index, edge_type=None, batch=None):
        """前向传播
        
        Args:
            x: 节点特征 [N, node_dim]
            edge_index: 边索引 [2, E]
            edge_type: 边类型 [E] (可选)
            batch: 批次信息 [N] (可选)
            
        Returns:
            节点嵌入 [N, hidden_dim]
        """
        # 节点嵌入
        h = self.node_embedding(x)
        
        # 边类型编码
        edge_attr = None
        if edge_type is not None:
            # 将边类型从{-1, 0, 1}映射到{0, 1, 2}
            edge_type_mapped = edge_type + 1  # -1->0, 0->1, 1->2
            edge_attr = self.edge_type_embedding(edge_type_mapped)
        
        # GPS层
        for layer in self.gps_layers:
            h = layer(h, edge_index, edge_attr, batch)
            
        return h
    
    def predict_edges(self, x, edge_index, edge_type=None, batch=None):
        """边预测
        
        Args:
            x: 节点特征 [N, node_dim]
            edge_index: 边索引 [2, E]
            edge_type: 边类型 [E] (可选)
            batch: 批次信息 [N] (可选)
            
        Returns:
            边预测结果 [E, 1]
        """
        h = self.forward(x, edge_index, edge_type, batch)
        
        # 边级预测
        src, dst = edge_index
        h_edge = torch.cat([h[src], h[dst]], dim=-1)
        edge_logits = self.edge_classifier(h_edge)
        
        return edge_logits
    
    def get_graph_embedding(self, x, edge_index, edge_type=None, batch=None):
        """获取图级嵌入"""
        h = self.forward(x, edge_index, edge_type, batch)
        
        if batch is not None:
            # 图级池化
            graph_emb = global_mean_pool(h, batch)
        else:
            # 简单平均池化
            graph_emb = h.mean(dim=0, keepdim=True)
            
        return graph_emb