from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def classification_metrics(y_true, y_pred, y_probs=None):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average='weighted')
    f1m = f1_score(y_true, y_pred, average='macro')
    pre_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    pre_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算AUC分数
    auc_weighted = None
    auc_macro = None
    
    if y_probs is not None:
        try:
            # 获取唯一类别数
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:
                # 二分类情况：使用正类概率
                auc_weighted = roc_auc_score(y_true, y_probs[:, 1])
                auc_macro = auc_weighted
            else:
                # 多分类情况：使用one-vs-rest策略
                auc_weighted = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
                auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception as e:
            # 如果AUC计算失败（例如只有一个类别），设为None
            print(f"Warning: AUC calculation failed: {e}")
            auc_weighted = None
            auc_macro = None
    
    result = {
        'acc': acc, 
        'weighted_f1': f1w, 
        'macro_f1': f1m, 
        'weighted_precision': pre_w, 
        'macro_precision': pre_m, 
        'weighted_recall': rec_w, 
        'macro_recall': rec_m
    }
    
    # 只有当AUC计算成功时才添加到结果中
    if auc_weighted is not None:
        result['weighted_auc'] = auc_weighted
    if auc_macro is not None:
        result['macro_auc'] = auc_macro
        
    return result

