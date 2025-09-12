import argparse
import json
from pathlib import Path

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/bitcoinalpha__seed0')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.55)  # 增加dropout以减少过拟合
    p.add_argument('--mpnn_type', type=str, default='gcn', choices=['gat', 'gcn', 'gin'], help='MPNN layer type')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--null_ratio', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=5e-4)  # 降低学习率以提高训练稳定性
    p.add_argument('--weight_decay', type=float, default=1e-2)  # 增加权重衰减以减少过拟合
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--patience', type=int, default=3, help='early stopping patience (epochs)')  # 增加耐心值以避免过早停止
    p.add_argument('--monitor', type=str, default='val_loss')
    p.add_argument('--seed', type=int, default=0)
    
    args = p.parse_args()

    # 从data_dir中提取数据集名称
    dataset_name = Path(args.data_dir).name
    extracted_dataset_name = dataset_name.split('__')[0] if '__' in dataset_name else dataset_name
    
    # 创建按数据集分类的保存目录
    dataset_save_dir = Path(args.save_dir) / extracted_dataset_name
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_save_dir/'config.json','w') as f:
        json.dump(vars(args), f, indent=2)
    return args