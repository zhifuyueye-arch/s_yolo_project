"""
依赖安装脚本
自动安装运行水面光伏锚固系统检测程序所需的所有依赖

使用方法:
    python install.py              # 安装基础依赖
    python install.py --full       # 安装完整依赖（包含可选）
    python install.py --upgrade    # 升级所有依赖

Author: CV Engineer
Date: 2026-02-27
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n📦 {description}...")
    print(f"   命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"   ✅ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 失败")
        print(f"   错误: {e.stderr}")
        return False


def install_requirements(full: bool = False, upgrade: bool = False):
    """安装requirements.txt中的依赖"""
    
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if upgrade:
        cmd.append("--upgrade")
    
    if full:
        # 安装完整依赖（包含可选）
        cmd.extend(["-r", "requirements.txt"])
        if run_command(cmd, "安装完整依赖"):
            print("\n✅ 完整依赖安装完成！")
            return True
    else:
        # 仅安装基础依赖（注释掉的除外）
        cmd.extend(["-r", "requirements.txt"])
        if run_command(cmd, "安装基础依赖"):
            print("\n✅ 基础依赖安装完成！")
            return True
    
    return False


def install_torch_cuda():
    """安装带CUDA支持的PyTorch"""
    print("\n" + "=" * 60)
    print("🎮 PyTorch CUDA 安装")
    print("=" * 60)
    print("\n请访问 https://pytorch.org/get-started/locally/ 选择适合您的版本")
    print("\n常用命令:")
    print("  CUDA 11.8:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n  CUDA 12.1:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n  CPU only:")
    print("    pip install torch torchvision")
    
    response = input("\n是否自动安装CUDA 11.8版本? (y/n): ").lower()
    
    if response == 'y':
        cmd = [
            sys.executable, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        return run_command(cmd, "安装 PyTorch + CUDA 11.8")
    
    return True


def verify_installation():
    """验证安装"""
    print("\n" + "=" * 60)
    print("🔍 验证安装")
    print("=" * 60)
    
    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics 未安装")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV 未安装")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy 未安装")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError:
        print("⚠️ Gradio 未安装 (Web UI将不可用)")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("   使用CPU模式")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="安装水面光伏锚固系统检测程序依赖",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python install.py              # 基础安装
  python install.py --full       # 完整安装
  python install.py --upgrade    # 升级依赖
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="安装完整依赖（包含可选功能）"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="升级已安装的依赖"
    )
    parser.add_argument(
        "--torch-cuda",
        action="store_true",
        help="安装带CUDA支持的PyTorch"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证安装，不执行安装"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 水面光伏锚固系统检测 - 依赖安装")
    print("=" * 60)
    
    # 仅验证
    if args.verify_only:
        if verify_installation():
            print("\n✅ 所有依赖已正确安装！")
            return 0
        else:
            print("\n❌ 部分依赖未安装")
            return 1
    
    # 检查pip
    print("\n📌 检查 pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip 可用")
    except:
        print("❌ pip 不可用，请安装Python并确保pip已配置")
        return 1
    
    # 升级pip
    print("\n📌 升级 pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                  capture_output=True)
    
    # 安装依赖
    if not install_requirements(full=args.full, upgrade=args.upgrade):
        print("\n❌ 依赖安装失败")
        return 1
    
    # 安装PyTorch CUDA（如果需要）
    if args.torch_cuda:
        install_torch_cuda()
    
    # 验证安装
    print("\n" + "=" * 60)
    if verify_installation():
        print("\n✅ 安装成功完成！")
        print("\n现在可以运行:")
        print("  python launch_ui.py     # 启动Web界面")
        print("  python check_env.py     # 检查环境")
        return 0
    else:
        print("\n⚠️ 安装可能不完整，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
