import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from gt_models import GraphTransformer
from utils.metrics import classification_metrics
from utils.early_stopping import EarlyStopping
from utils.logger import CombinedLogger
from data import read_posneg, read_null, iter_time_slices, build_data_from_df, scan_num_nodes
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss implementation for multi-class classification with class weights.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        alpha: Class weights tensor (num_classes,) or None for equal weights
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    
    Returns:
        Focal loss value
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    
    # Apply class weights if provided
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_t = alpha
        else:
            alpha_t = alpha.gather(0, targets)
        focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    else:
        focal_loss = (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def train_epoch_temporal(model, optimizer, df_train_posneg, num_nodes, device, null_ratio, batch_size, writer=None, epoch=0, args=None):
    model.train()
    losses = []
    y_all, p_all = [], []
    
    # 计算总batch数，基于真实边样本数量
    num_batches = (len(df_train_posneg) + batch_size - 1) // batch_size  # 向上取整
    
    batch_count = 0
    for batch_idx in range(num_batches):
        # 为每个batch选择batch_size个真实边样本
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df_train_posneg))
        actual_batch_size = end_idx - start_idx
        
        # 选择真实边样本（正样本）
        batch_real = df_train_posneg.iloc[start_idx:end_idx]
        batch_count += 1
        
        data = build_data_from_df(batch_real, num_nodes, device, 'train')
        real_ei, real_y = data.edge_index, data.y
        
        # 动态采样等量空边作为负样本
        null_ei = negative_sampling(edge_index=real_ei, num_nodes=num_nodes, num_neg_samples=actual_batch_size)
        null_y = torch.full((null_ei.size(1),), 2, dtype=torch.long, device=device)
        
        # 真实边和空边分离处理策略
        # 1. 先用真实边进行消息传递得到节点表示
        h_nodes = model.forward(data.x, data.edge_index, data.edge_type)
        
        # 2. 分别处理真实边样本和空边样本
        # 真实边样本的边预测
        src_real, dst_real = real_ei
        h_edge_real = torch.cat([h_nodes[src_real], h_nodes[dst_real]], dim=-1)
        logits_real = model.edge_classifier(h_edge_real)
        
        # 空边样本的边预测
        src_null, dst_null = null_ei
        h_edge_null = torch.cat([h_nodes[src_null], h_nodes[dst_null]], dim=-1)
        logits_null = model.edge_classifier(h_edge_null)
        
        # 3. 损失计算 - 使用Focal Loss解决类别不平衡问题
        # 设置类别权重：正边(0)=1.0, 负边(1)=2.0, 空边(2)=1.0
        class_weights = torch.tensor([1.0, 2.0, 1.0], device=device)
        # 真实边损失（包括正边和负边）
        loss_real = focal_loss(logits_real, real_y, alpha=class_weights, gamma=1.5)
        # 空边损失
        loss_null = focal_loss(logits_null, null_y, alpha=None, gamma=1.5)
        
        # 最终损失
        loss = loss_real + loss_null
        
        # 合并预测结果用于指标计算
        all_logits = torch.cat([logits_real, logits_null], dim=0)
        all_y = torch.cat([real_y, null_y], dim=0)
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        y_all.append(all_y.detach().cpu())
        p_all.append(all_logits.argmax(-1).detach().cpu())
        # 保存概率值用于AUC计算
        probs = F.softmax(all_logits, dim=-1).detach().cpu()
        if 'prob_all' not in locals():
            prob_all = []
        prob_all.append(probs)
    
    y = torch.cat(y_all).numpy()
    p = torch.cat(p_all).numpy()
    probs = torch.cat(prob_all).numpy()
    
    # 调试信息：打印最终结果的统计
    if epoch <= 3:
        print(f"[DEBUG] Epoch {epoch} final stats:")
        print(f"[DEBUG] Total batches processed: {batch_count}")
        print(f"[DEBUG] Total samples processed: {len(y)}")
        print(f"[DEBUG] True label distribution: {np.bincount(y)}")
        print(f"[DEBUG] Predicted label distribution: {np.bincount(p)}")
        print(f"[DEBUG] Average loss: {float(np.mean(losses)):.4f}")
    
    return float(np.mean(losses)), classification_metrics(y, p, probs)

@torch.no_grad()
def eval_epoch_temporal(model, df_posneg, df_null, num_nodes, device, batch_size, writer=None, epoch=0, args=None, null_ratio=1.0, split_type='val'):
    model.eval()
    losses = []
    y_all, p_all = [], []
    prob_all = []
    
    # 计算总batch数，基于真实边样本数量和空边样本数量的最小值
    min_samples = min(len(df_posneg), len(df_null))
    num_batches = (min_samples + batch_size - 1) // batch_size  # 向上取整
    
    # 调试信息：打印验证数据的基本信息
    if epoch <= 3:  # 只在前几个epoch打印调试信息
        print(f"[DEBUG] Epoch {epoch}, Split: {split_type}")
        print(f"[DEBUG] Real edge samples: {len(df_posneg)}, Null samples: {len(df_null)}")
        print(f"[DEBUG] Min samples: {min_samples}, Batch size: {batch_size}, Num batches: {num_batches}")
    
    batch_count = 0
    for batch_idx in range(num_batches):
        # 为每个batch选择batch_size个真实边样本和batch_size个空边样本
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, min_samples)
        actual_batch_size = end_idx - start_idx
        
        # 选择真实边样本
        if start_idx < len(df_posneg):
            batch_real = df_posneg.iloc[start_idx:start_idx + actual_batch_size]
        else:
            # 如果真实边样本不够，循环采样
            indices = np.arange(start_idx, start_idx + actual_batch_size) % len(df_posneg)
            batch_real = df_posneg.iloc[indices]
        
        # 选择对应数量的空边样本
        if start_idx < len(df_null):
            batch_null = df_null.iloc[start_idx:start_idx + actual_batch_size]
        else:
            # 如果空边样本不够，循环采样
            indices = np.arange(start_idx, start_idx + actual_batch_size) % len(df_null)
            batch_null = df_null.iloc[indices]
        
        batch_count += 1
        
        # 调试信息：打印每个batch的信息
        if epoch <= 3 and batch_count <= 3:  # 只打印前3个batch的信息
            print(f"[DEBUG] Batch {batch_count}: real={len(batch_real)}, null={len(batch_null)}")
        
        # 构建真实边数据
        data_real = build_data_from_df(batch_real, num_nodes, device, split_type)
        
        # 1. 先用真实边进行消息传递得到节点表示
        h_nodes = model.forward(data_real.x, data_real.edge_index, data_real.edge_type)
        
        # 2. 分别处理真实边样本和空边样本
        # 真实边样本的边预测
        src_real, dst_real = data_real.edge_index
        h_edge_real = torch.cat([h_nodes[src_real], h_nodes[dst_real]], dim=-1)
        logits_real = model.edge_classifier(h_edge_real)
        
        # 空边样本的边预测
        src_null = torch.tensor(batch_null['u'].to_numpy(), dtype=torch.long, device=device)
        dst_null = torch.tensor(batch_null['i'].to_numpy(), dtype=torch.long, device=device)
        h_edge_null = torch.cat([h_nodes[src_null], h_nodes[dst_null]], dim=-1)
        logits_null = model.edge_classifier(h_edge_null)
        
        # 3. 合并预测结果
        logits = torch.cat([logits_real, logits_null], dim=0)
        
        # 4. 构建真实标签
        null_labels = torch.full((len(batch_null),), 2, dtype=torch.long, device=device)
        all_y = torch.cat([data_real.y, null_labels], dim=0)
        
        # 5. 计算损失 - 验证/测试时使用普通交叉熵损失确保公平评估
        loss = F.cross_entropy(logits, all_y)
            
        losses.append(loss.item())
        
        y_all.append(all_y.detach().cpu())
        p_all.append(logits.argmax(-1).detach().cpu())
        # 保存概率值用于AUC计算
        probs = F.softmax(logits, dim=-1).detach().cpu()
        prob_all.append(probs)
    
    y = torch.cat(y_all).numpy()
    p = torch.cat(p_all).numpy()
    probs = torch.cat(prob_all).numpy()
    return float(np.mean(losses)), classification_metrics(y, p, probs)


def fit_and_evaluate(args):
    set_seed(args.seed)
    device = args.device
    dp = Path(args.save_dir); dp.mkdir(parents=True, exist_ok=True)

    # Paths
    train_posneg = Path(args.data_dir)/'train.csv'
    val_posneg   = Path(args.data_dir)/'val.csv'
    val_null     = Path(args.data_dir)/'val_neg.csv'
    test_posneg  = Path(args.data_dir)/'test.csv'
    test_null    = Path(args.data_dir)/'test_neg.csv'
    trd_posneg   = Path(args.data_dir)/'transductive_test.csv'
    trd_null     = Path(args.data_dir)/'transductive_test_neg.csv'
    ind_posneg   = Path(args.data_dir)/'inductive_test.csv'
    ind_null     = Path(args.data_dir)/'inductive_test_neg.csv'

    num_nodes = scan_num_nodes(((str(train_posneg), None), (str(val_posneg), str(val_null)), (str(test_posneg), str(test_null)), (str(trd_posneg), str(trd_null)), (str(ind_posneg), str(ind_null))))
    # DataFrames
    df_train = read_posneg(str(train_posneg))
    df_val_p = read_posneg(str(val_posneg))
    df_val_n = read_null(str(val_null))
    df_test_p = read_posneg(str(test_posneg))
    df_test_n = read_null(str(test_null))
    df_trd_p = read_posneg(str(trd_posneg))
    df_trd_n = read_null(str(trd_null))
    df_ind_p = read_posneg(str(ind_posneg))
    df_ind_n = read_null(str(ind_null))

    # Model & Opt
    model = GraphTransformer(
        node_dim=1,  # 使用节点ID作为特征
        hidden_dim=args.hidden_dim, 
        num_layers=args.layers, 
        heads=args.heads, 
        dropout=args.dropout,
        mpnn_type=args.mpnn_type
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 从data_dir中提取数据集名称
    dataset_name = Path(args.data_dir).name
    extracted_dataset_name = dataset_name.split('__')[0] if '__' in dataset_name else dataset_name
    
    # 创建按数据集分类的模型保存目录
    model_save_dir = Path("save_models") / extracted_dataset_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建按数据集分类的runs目录
    runs_save_dir = Path(args.save_dir) / extracted_dataset_name
    runs_save_dir.mkdir(parents=True, exist_ok=True)
    
    stopper = EarlyStopping(patience=args.patience, mode='max', monitor=args.monitor, save_dir=str(model_save_dir))
    logger = CombinedLogger(str(runs_save_dir), dataset_name)
    
    # 初始化TensorBoard writer
    tensorboard_dir = Path("runs") / extracted_dataset_name / f"exp_{args.seed}"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # 记录配置信息
    config_dict = {
        'epochs': args.epochs,
        'hidden_dim': args.hidden_dim,
        'layers': args.layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'null_ratio': args.null_ratio,
        'batch_size': args.batch_size,
        'device': args.device,
        'seed': args.seed
    }
    logger.log_config(config_dict)
    logger.log_message(f"Training started for {args.epochs} epochs", "INFO")
    logger.log_message(f"Save directory: {args.save_dir}", "INFO")
    logger.log_message(f"Device: {args.device}", "INFO")
    
    # Train loop
    for ep in range(1, args.epochs+1):
        tr_loss, tr_metrics = train_epoch_temporal(model, optimizer, df_train, num_nodes, device, args.null_ratio, args.batch_size, writer, ep, args)
        va_loss, va_metrics = eval_epoch_temporal(model, df_val_p, df_val_n, num_nodes, device, args.batch_size, writer, ep, args, args.null_ratio, split_type='val')
        row = {
            'epoch': ep,
            'train_loss': tr_loss,
            'train_acc': tr_metrics['acc'],
            'train_weighted_f1': tr_metrics['weighted_f1'],
            'train_macro_f1': tr_metrics['macro_f1'],
            'val_loss': va_loss,
            'val_acc': va_metrics['acc'],
            'val_weighted_f1': va_metrics['weighted_f1'],
            'val_macro_f1': va_metrics['macro_f1'],
        }
        # 使用新的logger记录epoch结果
        logger.log_epoch(
            epoch=ep,
            train_loss=tr_loss,
            train_acc=tr_metrics['acc'],
            val_loss=va_loss,
            val_acc=va_metrics['acc'],
            train_weighted_f1=tr_metrics['weighted_f1'],
            train_macro_f1=tr_metrics['macro_f1'],
            train_weighted_precision=tr_metrics['weighted_precision'],
            train_macro_precision=tr_metrics['macro_precision'],
            train_weighted_recall=tr_metrics['weighted_recall'],
            train_macro_recall=tr_metrics['macro_recall'],
            train_weighted_auc=tr_metrics.get('weighted_auc', 0.0),
            train_macro_auc=tr_metrics.get('macro_auc', 0.0),
            val_weighted_f1=va_metrics['weighted_f1'],
            val_macro_f1=va_metrics['macro_f1'],
            val_weighted_precision=va_metrics['weighted_precision'],
            val_macro_precision=va_metrics['macro_precision'],
            val_weighted_recall=va_metrics['weighted_recall'],
            val_macro_recall=va_metrics['macro_recall'],
            val_weighted_auc=va_metrics.get('weighted_auc', 0.0),
            val_macro_auc=va_metrics.get('macro_auc', 0.0)
        )
        
        # 记录TensorBoard指标
        # 损失
        writer.add_scalar('Loss/Train', tr_loss, ep)
        writer.add_scalar('Loss/Validation', va_loss, ep)
        
        # 准确率
        writer.add_scalar('Accuracy/Train', tr_metrics['acc'], ep)
        writer.add_scalar('Accuracy/Validation', va_metrics['acc'], ep)
        
        # F1分数
        writer.add_scalar('F1/Train_Weighted', tr_metrics['weighted_f1'], ep)
        writer.add_scalar('F1/Train_Macro', tr_metrics['macro_f1'], ep)
        writer.add_scalar('F1/Validation_Weighted', va_metrics['weighted_f1'], ep)
        writer.add_scalar('F1/Validation_Macro', va_metrics['macro_f1'], ep)
        
        # 精确率
        writer.add_scalar('Precision/Train_Weighted', tr_metrics['weighted_precision'], ep)
        writer.add_scalar('Precision/Train_Macro', tr_metrics['macro_precision'], ep)
        writer.add_scalar('Precision/Validation_Weighted', va_metrics['weighted_precision'], ep)
        writer.add_scalar('Precision/Validation_Macro', va_metrics['macro_precision'], ep)
        
        # 召回率
        writer.add_scalar('Recall/Train_Weighted', tr_metrics['weighted_recall'], ep)
        writer.add_scalar('Recall/Train_Macro', tr_metrics['macro_recall'], ep)
        writer.add_scalar('Recall/Validation_Weighted', va_metrics['weighted_recall'], ep)
        writer.add_scalar('Recall/Validation_Macro', va_metrics['macro_recall'], ep)
        
        # AUC（如果有的话）
        if 'weighted_auc' in tr_metrics:
            writer.add_scalar('AUC/Train_Weighted', tr_metrics['weighted_auc'], ep)
            writer.add_scalar('AUC/Train_Macro', tr_metrics['macro_auc'], ep)
        if 'weighted_auc' in va_metrics:
            writer.add_scalar('AUC/Validation_Weighted', va_metrics['weighted_auc'], ep)
            writer.add_scalar('AUC/Validation_Macro', va_metrics['macro_auc'], ep)
        
        # 计算监控指标
        monitor_value = va_metrics['macro_f1'] if args.monitor=='val_macro_f1' else (va_metrics['weighted_f1'] if args.monitor=='val_weighted_f1' else va_metrics['acc'])
        writer.add_scalar('Monitor/Value', monitor_value, ep)
        should_stop = stopper.step(monitor_value, model, ep)
        
        if should_stop:
            logger.log_message(f"Early stopping at epoch {ep} (monitor={args.monitor})", "INFO")
            logger.log_message(f"Best model saved at epoch {stopper.best_epoch} with {args.monitor}={stopper.best:.4f}", "INFO")
            break

    # Load best checkpoint
    stopper.load_best(model, map_location=device)

    # Final evaluations
    te_loss, te_metrics  = eval_epoch_temporal(model, df_test_p,  df_test_n, num_nodes, device, args.batch_size, args=args, split_type='test')
    trd_loss, trd_metrics = eval_epoch_temporal(model, df_trd_p,   df_trd_n,  num_nodes, device, args.batch_size, args=args, split_type='transductive_test')
    ind_loss, ind_metrics = eval_epoch_temporal(model, df_ind_p,   df_ind_n,  num_nodes, device, args.batch_size, args=args, split_type='inductive_test')

    results = {
        'test': te_metrics,
        'transductive_test': trd_metrics,
        'inductive_test': ind_metrics,
    }
    
    # 使用新的logger记录最终结果
    logger.log_message(f"Final results using best model from epoch {stopper.best_epoch}:", "INFO")
    logger.log_final_results(results)
    
    # 记录最终测试结果到TensorBoard
    final_epoch = stopper.best_epoch if hasattr(stopper, 'best_epoch') else ep
    
    # Test结果
    writer.add_scalar('Final/Test_Accuracy', te_metrics['acc'], final_epoch)
    writer.add_scalar('Final/Test_Weighted_F1', te_metrics['weighted_f1'], final_epoch)
    writer.add_scalar('Final/Test_Macro_F1', te_metrics['macro_f1'], final_epoch)
    writer.add_scalar('Final/Test_Loss', te_loss, final_epoch)
    
    # Transductive结果
    writer.add_scalar('Final/Transductive_Accuracy', trd_metrics['acc'], final_epoch)
    writer.add_scalar('Final/Transductive_Weighted_F1', trd_metrics['weighted_f1'], final_epoch)
    writer.add_scalar('Final/Transductive_Macro_F1', trd_metrics['macro_f1'], final_epoch)
    writer.add_scalar('Final/Transductive_Loss', trd_loss, final_epoch)
    
    # Inductive结果
    writer.add_scalar('Final/Inductive_Accuracy', ind_metrics['acc'], final_epoch)
    writer.add_scalar('Final/Inductive_Weighted_F1', ind_metrics['weighted_f1'], final_epoch)
    writer.add_scalar('Final/Inductive_Macro_F1', ind_metrics['macro_f1'], final_epoch)
    writer.add_scalar('Final/Inductive_Loss', ind_loss, final_epoch)
    
    # 保存详细的最终结果
    with open(runs_save_dir/'final_results.json','w') as f:
        json.dump(results, f, indent=2)
    
    # 保存简化的摘要信息
    summary = {
        'dataset': dataset_name,
        'epochs_trained': stopper.best_epoch if hasattr(stopper, 'best_epoch') else ep,
        'learning_rate': args.lr,
        'final_results': {
            'test_acc': te_metrics['acc'],
            'test_weighted_f1': te_metrics['weighted_f1'],
            'test_macro_f1': te_metrics['macro_f1'],
            'test_weighted_precision': te_metrics['weighted_precision'],
            'test_macro_precision': te_metrics['macro_precision'],
            'test_weighted_recall': te_metrics['weighted_recall'],
            'test_macro_recall': te_metrics['macro_recall'],
            'test_weighted_auc': te_metrics.get('weighted_auc', 0.0),
            'test_macro_auc': te_metrics.get('macro_auc', 0.0),
            'transductive_acc': trd_metrics['acc'],
            'transductive_weighted_f1': trd_metrics['weighted_f1'],
            'transductive_macro_f1': trd_metrics['macro_f1'],
            'transductive_weighted_precision': trd_metrics['weighted_precision'],
            'transductive_macro_precision': trd_metrics['macro_precision'],
            'transductive_weighted_recall': trd_metrics['weighted_recall'],
            'transductive_macro_recall': trd_metrics['macro_recall'],
            'transductive_weighted_auc': trd_metrics.get('weighted_auc', 0.0),
            'transductive_macro_auc': trd_metrics.get('macro_auc', 0.0),
            'inductive_acc': ind_metrics['acc'],
            'inductive_weighted_f1': ind_metrics['weighted_f1'],
            'inductive_macro_f1': ind_metrics['macro_f1'],
            'inductive_weighted_precision': ind_metrics['weighted_precision'],
            'inductive_macro_precision': ind_metrics['macro_precision'],
            'inductive_weighted_recall': ind_metrics['weighted_recall'],
            'inductive_macro_recall': ind_metrics['macro_recall'],
            'inductive_weighted_auc': ind_metrics.get('weighted_auc', 0.0),
            'inductive_macro_auc': ind_metrics.get('macro_auc', 0.0)
        },
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'layers': args.layers,
            'heads': args.heads,
            'dropout': args.dropout
        }
    }
    
    with open(runs_save_dir/'summary.json','w') as f:
        json.dump(summary, f, indent=2)

    # Save final model to save_models directory
    final_model_path = model_save_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_nodes': num_nodes,
        'hidden_dim': args.hidden_dim,
        'out_dim': 3,
        'num_layers': args.layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'results': results
    }, final_model_path)
    logger.log_message(f"Final model saved to: {final_model_path}", "INFO")
    
    logger.log_message("=== Final Results ===", "INFO")
    logger.log_message(f"[Test]           Acc={te_metrics['acc']:.4f} | F1w={te_metrics['weighted_f1']:.4f} | F1m={te_metrics['macro_f1']:.4f} | Pre_w={te_metrics['weighted_precision']:.4f} | Pre_m={te_metrics['macro_precision']:.4f} | Rec_w={te_metrics['weighted_recall']:.4f} | Rec_m={te_metrics['macro_recall']:.4f}", "INFO")
    logger.log_message(f"[Transductive]   Acc={trd_metrics['acc']:.4f} | F1w={trd_metrics['weighted_f1']:.4f} | F1m={trd_metrics['macro_f1']:.4f} | Pre_w={trd_metrics['weighted_precision']:.4f} | Pre_m={trd_metrics['macro_precision']:.4f} | Rec_w={trd_metrics['weighted_recall']:.4f} | Rec_m={trd_metrics['macro_recall']:.4f}", "INFO")
    logger.log_message(f"[Inductive]      Acc={ind_metrics['acc']:.4f} | F1w={ind_metrics['weighted_f1']:.4f} | F1m={ind_metrics['macro_f1']:.4f} | Pre_w={ind_metrics['weighted_precision']:.4f} | Pre_m={ind_metrics['macro_precision']:.4f} | Rec_w={ind_metrics['weighted_recall']:.4f} | Rec_m={ind_metrics['macro_recall']:.4f}", "INFO")
    
    logger.log_message("Training completed successfully!", "INFO")
    
    # 关闭TensorBoard writer
    writer.close()
    
    # 确保日志文件被正确关闭和刷新
    logger.close()
