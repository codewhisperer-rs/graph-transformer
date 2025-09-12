#!/usr/bin/env python3
"""
启动TensorBoard服务器的脚本
"""
import subprocess
import sys
from pathlib import Path

def start_tensorboard(logdir="runs", port=6006):
    """启动TensorBoard服务器"""
    logdir_path = Path(logdir)
    
    if not logdir_path.exists():
        print(f"警告：日志目录 {logdir_path.absolute()} 不存在")
        print("请先运行训练脚本生成日志文件")
        return
    
    print(f"启动TensorBoard服务器...")
    print(f"日志目录: {logdir_path.absolute()}")
    print(f"端口: {port}")
    print(f"访问地址: http://localhost:{port}")
    print("按 Ctrl+C 停止服务器")
    
    try:
        # 启动TensorBoard
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port), "--bind_all"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard服务器已停止")
    except FileNotFoundError:
        print("错误：找不到tensorboard命令")
        print("请安装tensorboard: pip install tensorboard")
    except Exception as e:
        print(f"启动TensorBoard时发生错误: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动TensorBoard服务器")
    parser.add_argument("--logdir", default="runs", help="TensorBoard日志目录 (默认: runs)")
    parser.add_argument("--port", type=int, default=6006, help="端口号 (默认: 6006)")
    
    args = parser.parse_args()
    start_tensorboard(args.logdir, args.port)
