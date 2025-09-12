from typing import Optional, Iterator, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data

# --- CSV Readers ---

def read_posneg(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"u","i","ts","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {need}")
    # 确保数据按时间排序，避免时间泄漏
    return df[["u","i","ts","label"]].sort_values('ts', kind='mergesort').reset_index(drop=True).copy()

def read_null(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"u","i","ts"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {need}")
    out = df[["u","i","ts"]].copy()
    out["label"] = 2
    # 确保数据按时间排序，避免时间泄漏
    return out.sort_values('ts', kind='mergesort').reset_index(drop=True)

# --- Utilities ---

def label_to_class(sr: pd.Series) -> pd.Series:
    mapping = {1:0, -1:1, 2:2}
    return sr.map(mapping)

def scan_num_nodes(paths: Tuple[Tuple[str, Optional[str]], ...]) -> int:
    max_u_id = 0  # 用户节点最大ID
    max_i_id = 0  # 物品节点最大ID
    for posneg_path, null_path in paths:
        dfp = read_posneg(posneg_path)
        max_u_id = max(max_u_id, int(dfp['u'].max()))
        max_i_id = max(max_i_id, int(dfp['i'].max()))
        if null_path is not None:
            dfn = read_null(null_path)
            max_u_id = max(max_u_id, int(dfn['u'].max()))
            max_i_id = max(max_i_id, int(dfn['i'].max()))
    # 二部图总节点数 = 用户节点数 + 物品节点数
    # 用户节点: 0 到 max_u_id (共 max_u_id + 1 个)
    # 物品节点: (max_u_id + 1) 到 (max_u_id + 1 + max_i_id) (共 max_i_id + 1 个)
    return (max_u_id + 1) + (max_i_id + 1)

def iter_time_slices(df: pd.DataFrame, edges_per_batch: int = None, seconds_per_batch: int = None) -> Iterator[pd.DataFrame]:
    assert edges_per_batch is not None or seconds_per_batch is not None
    df = df.sort_values('ts', kind='mergesort').reset_index(drop=True)
    if edges_per_batch is not None:
        for start in range(0, len(df), edges_per_batch):
            yield df.iloc[start:start+edges_per_batch]
    else:
        # simple rolling window by seconds
        start = 0
        while start < len(df):
            t0 = df['ts'].iloc[start]
            mask = df['ts'] < t0 + seconds_per_batch
            end = mask.values.nonzero()[0][-1] + 1 if mask.any() else start+1
            yield df.iloc[start:end]
            start = end

def build_data_from_df(df: pd.DataFrame, num_nodes: int, device: str, split_type: str = 'train') -> Data:
    import numpy as np
    src = torch.tensor(df['u'].to_numpy(np.int64), dtype=torch.long)
    dst = torch.tensor(df['i'].to_numpy(np.int64), dtype=torch.long)
    dst += int(src.max()) + 1   # To make graph bipartite, in case of bipartite graphs
    y = torch.tensor(label_to_class(df['label']).to_numpy(np.int64), dtype=torch.long)
    # edge_type应该是原始标签（-1,0,1），不是分类后的标签
    edge_type = torch.tensor(df['label'].to_numpy(np.int64), dtype=torch.long)
    # 将null样本的edge_type设为0（中性）
    edge_type = torch.where(edge_type == 2, 0, edge_type)
    edge_index = torch.stack([src, dst], dim=0)
    
    # 为不同数据集分割使用不同的随机种子，避免数据泄漏
    seed_map = {'train': 42, 'val': 43, 'test': 44, 'transductive_test': 45, 'inductive_test': 46}
    torch.manual_seed(seed_map.get(split_type, 42))
    node_features = torch.randn(num_nodes, 1, dtype=torch.float)  # [num_nodes, 1]
    return Data(x=node_features, edge_index=edge_index, edge_type=edge_type, y=y).to(device)
