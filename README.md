# TensorBoard 监控训练过程

## 功能说明

已经在训练脚本中添加了完整的TensorBoard支持，可以实时监控以下指标：

### 训练过程监控
- **损失 (Loss)**：训练损失和验证损失
- **准确率 (Accuracy)**：训练和验证准确率
- **F1分数 (F1 Score)**：加权F1和宏平均F1（训练和验证）
- **精确率 (Precision)**：加权精确率和宏平均精确率（训练和验证）
- **召回率 (Recall)**：加权召回率和宏平均召回率（训练和验证）
- **AUC**：如果可用，显示AUC指标
- **监控指标**：显示早停监控的指标值

### 最终测试结果
- **Test结果**：最终测试集的性能指标
- **Transductive结果**：转导学习测试结果
- **Inductive结果**：归纳学习测试结果

## 使用方法

### 1. 运行训练
```bash
python train.py --data_dir /path/to/dataset --epochs 100
```

### 2. 启动TensorBoard
```bash
# 方法1：使用提供的脚本
python start_tensorboard.py

# 方法2：直接使用tensorboard命令
tensorboard --logdir runs --port 6006

# 方法3：指定特定数据集的日志
tensorboard --logdir runs/bitcoinalpha --port 6006
```

### 3. 查看结果
在浏览器中打开 http://localhost:6006

## 日志目录结构

```
runs/
├── bitcoinalpha/
│   └── exp_0/         # seed=0的实验
├── epinions/
│   └── exp_0/
└── wiki-RfA/
    └── exp_0/
```

每个数据集的日志都会保存在单独的目录中，方便比较不同数据集的训练效果。

## 主要改进

1. **组织化的日志结构**：按数据集和实验种子分类保存日志
2. **全面的指标监控**：涵盖所有重要的分类指标
3. **实时监控**：每个epoch都会更新指标
4. **最终结果记录**：训练完成后记录所有测试结果
5. **便捷的启动脚本**：一键启动TensorBoard服务

## 注意事项

- TensorBoard日志保存在 `runs/` 目录中
- 每次运行会根据数据集名称和随机种子创建新的日志目录
- 如果端口6006被占用，可以使用 `--port` 参数指定其他端口
- 训练过程中可以实时查看TensorBoard，无需等待训练完成
