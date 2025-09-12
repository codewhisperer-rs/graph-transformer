from pathlib import Path
import csv
import datetime
import os

class CSVLogger:
    def __init__(self, save_dir, filename='metrics.csv'):
        self.path = Path(save_dir)/filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, 'a', newline='')
        self.writer = csv.writer(self.file)

    def log(self, **kwargs):
        if not self._header_written:
            self.writer.writerow(list(kwargs.keys()))
            self._header_written = True
        self.writer.writerow(list(kwargs.values()))
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

class DatasetLogger:
    """按数据集分类的日志记录器，使用时间戳命名txt文件"""
    
    def __init__(self, base_log_dir="logs", dataset_name=None):
        self.base_log_dir = Path(base_log_dir)
        # 自动提取数据集名称，去掉__seed0等后缀
        if dataset_name:
            self.dataset_name = self._extract_dataset_name(dataset_name)
        else:
            self.dataset_name = None
        self.start_time = datetime.datetime.now()
        self.log_file = None
        self.log_path = None
        
        if self.dataset_name:
            self._setup_log_file()
    
    def _extract_dataset_name(self, full_name):
        """提取数据集名称，去掉__seed0等后缀"""
        # 如果包含__，则取第一部分
        if '__' in full_name:
            return full_name.split('__')[0]
        return full_name
    
    def _setup_log_file(self):
        """设置日志文件路径和创建目录"""
        # 创建数据集文件夹
        dataset_dir = self.base_log_dir / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.txt"
        self.log_path = dataset_dir / filename
        
        # 打开日志文件
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # 写入开始信息
        self.log_file.write(f"=== 训练日志 ===\n")
        self.log_file.write(f"数据集: {self.dataset_name}\n")
        self.log_file.write(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"="*50 + "\n\n")
        self.log_file.flush()
    
    def log(self, message, level="INFO"):
        """记录日志消息"""
        if not self.log_file or self.log_file.closed:
            print(f"[{level}] {message}")
            return
            
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_file.write(log_entry)
        self.log_file.flush()
        
        # 同时输出到控制台
        print(f"[{timestamp}] [{level}] {message}")
    
    def log_config(self, config_dict):
        """记录配置信息"""
        self.log("=== 配置信息 ===")
        for key, value in config_dict.items():
            self.log(f"{key}: {value}")
        self.log("=" * 30)
    
    def log_message(self, message, level="INFO"):
        """记录普通消息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        if self.log_file:
            self.log_file.write(log_entry)
            self.log_file.flush()
        
        # 同时输出到控制台
        print(f"[{timestamp}] [{level}] {message}")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_acc, val_loss=None, **kwargs):
        """记录epoch训练结果"""
        # 创建表格化的输出格式
        self.log(f"--- Epoch {epoch} ---")
        
        # 表头
        header = f"{'Train':<35} | {'Validation':<35}"
        separator = f"{'-'*35} | {'-'*35}"
        
        self.log(header)
        self.log(separator)
        
        # Loss (训练和验证)
        if val_loss is not None:
            loss_line = f"Loss: {train_loss:.4f}{'':<23} | Loss: {val_loss:.4f}"
        else:
            loss_line = f"Loss: {train_loss:.4f}{'':<23} |"
        self.log(loss_line)
        
        # 准确率
        acc_line = f"Acc: {train_acc:.4f}{'':<24} | Acc: {val_acc:.4f}"
        self.log(acc_line)
        
        # 处理其他配对指标
        train_metrics = {}
        val_metrics = {}
        
        for key, value in kwargs.items():
            if key.startswith('train_'):
                metric_name = key[6:]  # 去掉'train_'前缀
                train_metrics[metric_name] = value
            elif key.startswith('val_'):
                metric_name = key[4:]   # 去掉'val_'前缀
                val_metrics[metric_name] = value
        
        # 显示配对的指标
        metric_names = {
            'weighted_f1': 'Weighted F1',
            'macro_f1': 'Macro F1', 
            'weighted_precision': 'Weighted Precision',
            'macro_precision': 'Macro Precision',
            'weighted_recall': 'Weighted Recall',
            'macro_recall': 'Macro Recall',
            'weighted_auc': 'Weighted AUC',
            'macro_auc': 'Macro AUC'
        }
        
        for metric_key, display_name in metric_names.items():
            if metric_key in train_metrics and metric_key in val_metrics:
                train_val = train_metrics[metric_key]
                val_val = val_metrics[metric_key]
                if isinstance(train_val, float) and isinstance(val_val, float):
                    # 计算左侧需要的空格数，确保对齐到35个字符
                    train_text = f"{display_name}: {train_val:.4f}"
                    val_text = f"{display_name}: {val_val:.4f}"
                    padding = 35 - len(train_text)
                    line = f"{train_text}{' ' * padding} | {val_text}"
                    self.log(line)
        
        self.log("")  # 空行分隔
    
    def log_final_results(self, results_dict):
        """记录最终结果"""
        self.log("\n=== 最终结果 ===")
        
        # 详细记录每个测试集的结果
        for test_type, metrics in results_dict.items():
            if test_type == 'test':
                test_name = "Test"
            elif test_type == 'transductive_test':
                test_name = "Transductive"
            elif test_type == 'inductive_test':
                test_name = "Inductive"
            else:
                test_name = test_type.title()
            
            if isinstance(metrics, dict):
                self.log(f"\n[{test_name}]")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        if metric_name == 'acc':
                            self.log(f"  准确率 (Accuracy): {metric_value:.4f}")
                        elif metric_name == 'weighted_f1':
                            self.log(f"  加权F1 (Weighted F1): {metric_value:.4f}")
                        elif metric_name == 'macro_f1':
                            self.log(f"  宏F1 (Macro F1): {metric_value:.4f}")
                        elif metric_name == 'weighted_precision':
                            self.log(f"  加权精确率 (Weighted Precision): {metric_value:.4f}")
                        elif metric_name == 'macro_precision':
                            self.log(f"  宏精确率 (Macro Precision): {metric_value:.4f}")
                        elif metric_name == 'weighted_recall':
                            self.log(f"  加权召回率 (Weighted Recall): {metric_value:.4f}")
                        elif metric_name == 'macro_recall':
                            self.log(f"  宏召回率 (Macro Recall): {metric_value:.4f}")
                        elif metric_name == 'weighted_auc':
                            self.log(f"  加权AUC (Weighted AUC): {metric_value:.4f}")
                        elif metric_name == 'macro_auc':
                            self.log(f"  宏AUC (Macro AUC): {metric_value:.4f}")
                        else:
                            self.log(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        self.log(f"  {metric_name}: {metric_value}")
            else:
                if isinstance(metrics, float):
                    self.log(f"{test_name}: {metrics:.4f}")
                else:
                    self.log(f"{test_name}: {metrics}")
        
        # 添加汇总信息
        self.log("\n=== 结果汇总 ===")
        for test_type, metrics in results_dict.items():
            if isinstance(metrics, dict) and 'acc' in metrics:
                test_name = test_type.replace('_', ' ').title()
                acc = metrics.get('acc', 0)
                f1w = metrics.get('weighted_f1', 0)
                f1m = metrics.get('macro_f1', 0)
                self.log(f"[{test_name:<12}] Acc={acc:.4f} | F1w={f1w:.4f} | F1m={f1m:.4f}")
        
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        self.log(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"总耗时: {duration}")
        self.log("=" * 50)
        
        # 确保所有内容都被写入文件
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                print(f"关闭日志文件时出错: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class CombinedLogger:
    """日志记录器，支持文本日志"""
    
    def __init__(self, save_dir, dataset_name, csv_filename='metrics.csv'):
        # 日志文件放在logs文件夹中
        log_dir = "logs"
        self.dataset_logger = DatasetLogger(log_dir, dataset_name)
    
    def _extract_dataset_name(self, full_name):
        """提取数据集名称，去掉__seed0等后缀"""
        if '__' in full_name:
            return full_name.split('__')[0]
        return full_name
    
    def log_metrics(self, **kwargs):
        """记录指标（已禁用CSV功能）"""
        pass
    
    def log_message(self, message, level="INFO"):
        """记录消息到文本日志"""
        self.dataset_logger.log(message, level)
    
    def log_config(self, config_dict):
        """记录配置信息"""
        self.dataset_logger.log_config(config_dict)
    
    def log_epoch(self, epoch, train_loss, train_acc, val_acc, **kwargs):
        """记录到文本日志"""
        # 只记录到文本日志
        self.dataset_logger.log_epoch(epoch, train_loss, train_acc, val_acc, **kwargs)
    
    def log_final_results(self, results_dict):
        """记录最终结果"""
        self.dataset_logger.log_final_results(results_dict)
    
    def close(self):
        """关闭所有日志"""
        self.dataset_logger.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()