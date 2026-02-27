"""
环境检查脚本
检查是否满足运行水面光伏锚固系统检测程序的所有依赖

使用方法:
    python check_env.py

Author: CV Engineer
Date: 2026-02-27
"""

import sys
import subprocess
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)"


def check_package(package_name: str, import_name: str = None, min_version: str = None) -> Tuple[bool, str]:
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # 简单版本比较
            if version.split('.')[0] >= min_version.split('.')[0]:
                return True, f"✅ {package_name} {version}"
            else:
                return False, f"⚠️ {package_name} {version} (建议 >= {min_version})"
        
        return True, f"✅ {package_name} {version}"
    except ImportError:
        return False, f"❌ {package_name} (未安装)"


def check_cuda() -> Tuple[bool, str]:
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"✅ CUDA {cuda_version} | {device_count} 个设备 | {device_name}"
        else:
            return False, "⚠️ CUDA 不可用 (将使用CPU运行，速度较慢)"
    except ImportError:
        return False, "❌ PyTorch 未安装"


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 环境检查 - 水面光伏锚固系统检测")
    print("=" * 60)
    
    all_checks = []
    
    # 1. 检查Python版本
    print("\n📌 Python 环境")
    print("-" * 40)
    ok, msg = check_python_version()
    all_checks.append(ok)
    print(f"  {msg}")
    
    # 2. 检查核心依赖
    print("\n📌 核心依赖")
    print("-" * 40)
    core_packages = [
        ('ultralytics', 'ultralytics', '8.1.0'),
        ('opencv-python', 'cv2', '4.8.0'),
        ('numpy', 'numpy', '1.24.0'),
        ('pandas', 'pandas', '2.0.0'),
        ('matplotlib', 'matplotlib', '3.7.0'),
        ('Pillow', 'PIL', '10.0.0'),
        ('PyYAML', 'yaml', '6.0'),
        ('tqdm', 'tqdm', '4.65.0'),
        ('psutil', 'psutil', '5.9.0'),
    ]
    
    for pkg_name, import_name, min_ver in core_packages:
        ok, msg = check_package(pkg_name, import_name, min_ver)
        all_checks.append(ok)
        print(f"  {msg}")
    
    # 3. 检查Web UI依赖
    print("\n📌 Web UI 依赖")
    print("-" * 40)
    ui_packages = [
        ('gradio', 'gradio', '4.0.0'),
    ]
    
    for pkg_name, import_name, min_ver in ui_packages:
        ok, msg = check_package(pkg_name, import_name, min_ver)
        all_checks.append(ok)
        print(f"  {msg}")
    
    # 4. 检查可选依赖
    print("\n📌 可选依赖（用于高级功能）")
    print("-" * 40)
    optional_packages = [
        ('seaborn', 'seaborn', '0.12.0'),
        ('plotly', 'plotly', '5.18.0'),
        ('onnx', 'onnx', '1.14.0'),
        ('onnxruntime', 'onnxruntime', '1.16.0'),
    ]
    
    for pkg_name, import_name, min_ver in optional_packages:
        ok, msg = check_package(pkg_name, import_name, min_ver)
        # 可选依赖不影响总体结果
        print(f"  {msg}")
    
    # 5. 检查CUDA
    print("\n📌 CUDA / GPU 支持")
    print("-" * 40)
    ok, msg = check_cuda()
    all_checks.append(ok)  # CUDA是可选但推荐的
    print(f"  {msg}")
    
    # 6. 总结
    print("\n" + "=" * 60)
    print("📊 检查结果")
    print("=" * 60)
    
    core_ok = all(all_checks[:len(core_packages)+2])  # Python + 核心依赖
    ui_ok = all_checks[len(core_packages)+2]  # Gradio
    
    if core_ok and ui_ok:
        print("✅ 所有必要依赖已安装，可以运行所有功能！")
        print("\n启动命令:")
        print("  python launch_ui.py          # 启动Web界面")
        print("  python data_prep_ui.py       # 仅启动数据预处理UI")
        print("  python training_ui.py        # 仅启动训练UI")
    elif core_ok:
        print("✅ 核心功能可用，但Web UI需要安装Gradio:")
        print("  pip install gradio>=4.0.0")
        print("\n或运行命令行版本:")
        print("  python data_prep/dataset_processor.py")
        print("  python training/train_models.py")
    else:
        print("❌ 缺少必要依赖，请安装:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    # 返回退出码
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
