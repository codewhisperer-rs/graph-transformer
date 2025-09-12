import torch
from pathlib import Path
from gt_models import GraphTransformer

def load_model_from_pth(model_path, device='cpu'):
    """
    从.pth文件加载训练好的GraphTransformer模型
    
    Args:
        model_path (str): .pth模型文件路径
        device (str): 设备类型 ('cpu' 或 'cuda')
    
    Returns:
        model: 加载好的GraphTransformer模型
        model_info: 包含模型配置和结果的字典
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    model = GraphTransformer(
        num_nodes=checkpoint['num_nodes'],
        hidden_dim=checkpoint['hidden_dim'],
        out_dim=checkpoint['out_dim'],
        num_layers=checkpoint['num_layers'],
        heads=checkpoint['heads'],
        dropout=checkpoint['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 提取模型信息
    model_info = {
        'num_nodes': checkpoint['num_nodes'],
        'hidden_dim': checkpoint['hidden_dim'],
        'out_dim': checkpoint['out_dim'],
        'num_layers': checkpoint['num_layers'],
        'heads': checkpoint['heads'],
        'dropout': checkpoint['dropout'],
        'results': checkpoint.get('results', None)
    }
    
    return model, model_info

def load_best_model_from_dir(save_dir, device='cpu'):
    """
    从保存目录加载最佳模型 (best.pth)
    
    Args:
        save_dir (str): 模型保存目录
        device (str): 设备类型 ('cpu' 或 'cuda')
    
    Returns:
        model: 加载好的GraphTransformer模型
        epoch: 最佳模型的训练轮次
    """
    best_model_path = Path(save_dir) / 'best.pth'
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # 注意：best.pth只包含state_dict和epoch，需要从其他地方获取模型配置
    # 这里假设有config.json文件包含模型配置
    import json
    config_path = Path(save_dir) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 从数据目录扫描节点数量
        from data import scan_num_nodes, read_posneg
        train_path = Path(config['data_dir']) / 'train.csv'
        val_path = Path(config['data_dir']) / 'val.csv'
        val_null_path = Path(config['data_dir']) / 'val_neg.csv'
        test_path = Path(config['data_dir']) / 'test.csv'
        test_null_path = Path(config['data_dir']) / 'test_neg.csv'
        trd_path = Path(config['data_dir']) / 'transductive_test.csv'
        trd_null_path = Path(config['data_dir']) / 'transductive_test_neg.csv'
        ind_path = Path(config['data_dir']) / 'inductive_test.csv'
        ind_null_path = Path(config['data_dir']) / 'inductive_test_neg.csv'
        
        num_nodes = scan_num_nodes((
            (str(train_path), None),
            (str(val_path), str(val_null_path)),
            (str(test_path), str(test_null_path)),
            (str(trd_path), str(trd_null_path)),
            (str(ind_path), str(ind_null_path))
        ))
        
        model = GraphTransformer(
            num_nodes=num_nodes,
            hidden_dim=config['hidden_dim'],
            out_dim=3,
            num_layers=config['layers'],
            heads=config['heads'],
            dropout=config['dropout']
        )
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}. Cannot determine model architecture.")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('epoch', None)

# 使用示例
if __name__ == "__main__":
    # 加载最终模型
    # model, info = load_model_from_pth('runs/exp/final_model.pth', device='cuda')
    # print(f"Model loaded with {info['num_nodes']} nodes, {info['num_layers']} layers")
    # print(f"Test results: {info['results']['test']}")
    
    # 加载最佳模型
    # model, epoch = load_best_model_from_dir('runs/exp', device='cuda')
    # print(f"Best model from epoch {epoch} loaded")
    pass