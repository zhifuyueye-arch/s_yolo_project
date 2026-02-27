"""
水面光伏锚固系统 - 统一UI启动器
Main Launcher for Web UI

使用方式:
    python launch_ui.py          # 启动所有服务
    python launch_ui.py --data   # 仅启动数据预处理UI
    python launch_ui.py --train  # 仅启动训练UI
    
依赖:
    pip install gradio

Author: CV Engineer
Date: 2026-02-27
"""

import os
import sys
import argparse
import subprocess
import time
import webbrowser
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🌊 水面光伏锚固系统智能检测 - Web UI 启动器 🌊            ║
    ║                                                              ║
    ║     Floating PV Anchor Detection System                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import gradio
        print("✅ Gradio 已安装")
        return True
    except ImportError:
        print("❌ Gradio 未安装")
        print("请运行: pip install gradio")
        return False


def launch_data_ui():
    """启动数据预处理UI"""
    print("\n📁 启动数据预处理 UI...")
    print("   地址: http://127.0.0.1:7860")
    
    try:
        import data_prep_ui
        data_prep_ui.main()
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def launch_train_ui():
    """启动训练UI"""
    print("\n🚀 启动模型训练 UI...")
    print("   地址: http://127.0.0.1:7861")
    
    try:
        import training_ui
        training_ui.main()
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def launch_both():
    """同时启动两个UI（使用子进程）"""
    print("\n🔄 启动所有服务...")
    
    processes = []
    
    # 启动数据预处理UI
    print("📁 启动数据预处理 UI (端口 7860)...")
    p1 = subprocess.Popen([sys.executable, "data_prep_ui.py"])
    processes.append(("Data Prep", p1, 7860))
    
    # 等待第一个服务启动
    time.sleep(3)
    
    # 启动训练UI
    print("🚀 启动模型训练 UI (端口 7861)...")
    p2 = subprocess.Popen([sys.executable, "training_ui.py"])
    processes.append(("Training", p2, 7861))
    
    # 等待服务启动
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("✅ 所有服务已启动!")
    print("=" * 60)
    print("📁 数据预处理 UI: http://127.0.0.1:7860")
    print("🚀 模型训练 UI:   http://127.0.0.1:7861")
    print("=" * 60)
    print("按 Ctrl+C 停止所有服务\n")
    
    # 尝试自动打开浏览器
    try:
        webbrowser.open("http://127.0.0.1:7860")
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:7861")
    except:
        pass
    
    # 等待进程
    try:
        for name, process, port in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n\n正在停止所有服务...")
        for name, process, port in processes:
            process.terminate()
            print(f"  已停止 {name} (端口 {port})")
        print("所有服务已停止")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="水面光伏锚固系统 - UI启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python launch_ui.py          # 启动所有服务
  python launch_ui.py --data   # 仅启动数据预处理UI
  python launch_ui.py --train  # 仅启动训练UI
        """
    )
    
    parser.add_argument(
        "--data",
        action="store_true",
        help="仅启动数据预处理UI"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="仅启动模型训练UI"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 根据参数启动
    if args.data:
        launch_data_ui()
    elif args.train:
        launch_train_ui()
    else:
        launch_both()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
