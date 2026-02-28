# 水面光伏区锚固系统智能检测系统

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8/v12-green.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

基于YOLOv8和YOLOv12的水面光伏区锚固系统组件与缺陷智能检测系统。

## 📋 项目概述

本项目针对水面光伏区的锚固系统进行自动化检测，利用无人机或岸边摄像头采集的图像，识别锚绳、连接件、锚块、浮筒连接件等组件的状态（正常、松动、断裂/缺失、破损）。

### 检测类别 (16类)

| 部件 | 正常 | 松动/磨损 | 断裂/缺失 | 破损/腐蚀 |
|------|-----|----------|----------|----------|
| 锚绳 | ✅ | ✅ | ✅ | ✅ |
| 连接件 | ✅ | ✅ | ✅ | ✅ |
| 锚块 | ✅ | ✅ | ✅ | ✅ |
| 浮筒连接件 | ✅ | ✅ | ✅ | ✅ |

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-org/anchor-detection.git
cd anchor-detection

# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 方式1: 使用安装脚本（推荐）
python install.py

# 方式2: 手动安装
pip install -r requirements.txt

# 验证安装
python check_env.py
```

## 🖥️ Web UI 使用（推荐）

项目提供可视化的Web界面，无需编写代码即可完成数据处理和模型训练。

### 启动Web界面

```bash
# 启动所有服务（数据预处理 + 模型训练）
python launch_ui.py

# 或单独启动
python data_prep_ui.py    # 数据预处理界面 (端口 7860)
python training_ui.py     # 模型训练界面 (端口 7861)
```

启动后自动打开浏览器，访问：
- 📁 **数据预处理**: http://127.0.0.1:7860
- 🚀 **模型训练**: http://127.0.0.1:7861

### 界面功能

**数据预处理界面**
- 配置输入/输出路径
- 设置数据集划分比例（训练/验证/测试）
- 调整数据增强参数（水面波纹、反光、藻类模拟）
- 实时查看处理日志

**模型训练界面**
- 选择模型（YOLOv8n / YOLOv12）
- 配置训练参数（轮数、批次、学习率等）
- 模型性能对比分析
- 一键导出多种格式（ONNX/TensorRT/OpenVINO）

---

## ⌨️ 命令行使用

如需使用命令行操作，请参考以下步骤：

### 数据准备

```bash
# 1. 准备原始数据
# 将图像和标注文件放入 data/raw/ 目录
# 支持的标注格式: JSON, XML

# 2. 运行数据预处理
python data_prep/dataset_processor.py

# 输出目录: data/processed/
# - images/train, val, test
# - labels/train, val, test
# - data.yaml
```

### 模型训练

```bash
# 运行完整训练流程 (YOLOv8n + YOLOv12对比)
python training/train_models.py

# 输出目录: results/
# - models/ - 训练好的权重
# - evaluation/ - 评估报告和可视化
```

## 📁 项目结构

```
project_root/
├── data_prep_ui.py            # ⭐ Web UI: 数据预处理界面 (端口7860)
├── training_ui.py             # ⭐ Web UI: 模型训练界面 (端口7861)
├── launch_ui.py               # ⭐ Web UI: 统一启动器
├── check_env.py               # 环境检查工具
├── install.py                 # 依赖安装脚本
│
├── data_prep/                  # 模块A: 数据集构建与预处理
│   ├── dataset_processor.py    # 核心数据处理脚本
│   └── __init__.py
│
├── training/                   # 模块B: 模型训练与评估
│   ├── train_models.py         # 训练主脚本
│   └── __init__.py
│
├── configs/                    # 配置文件
│   ├── data.yaml              # 数据集配置
│   └── hyp.yaml               # 超参数配置
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
│
├── results/                    # 输出结果
│   ├── models/                # 模型权重
│   ├── evaluation/            # 评估报告
│   └── visualized_dataset/    # 可视化样本
│
├── logs/                       # 日志文件
├── requirements.txt            # 依赖列表
├── .gitignore                 # Git忽略配置
├── GIT_WORKFLOW.md            # Git工作流指南
├── OPTIMIZATION_GUIDE.md      # 优化建议
└── README.md                  # 项目说明
```

## 🔧 模块说明

### 模块A: 数据集构建与预处理

**功能特性：**
- 支持JSON/XML标注格式解析
- 数据清洗（分辨率过滤、目标尺寸过滤）
- 水面场景专用增强（波纹、反光、藻类模拟）
- Mosaic/Mixup增强
- 自动数据集划分（7:2:1）
- 标注可视化

**使用示例：**

```python
from data_prep.dataset_processor import DatasetProcessor

processor = DatasetProcessor(
    raw_data_dir='data/raw',
    output_dir='data/processed',
    min_resolution=(1920, 1080),
    min_target_size=50,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)

# 执行处理
processor.process_dataset()

# 生成可视化
processor.visualize_dataset(num_samples=20)
```

### 模块B: 模型训练与对比评估

**功能特性：**
- YOLOv8n Baseline训练（边缘部署）
- YOLOv12 SOTA训练（高精度）
- 自动混合精度 (AMP)
- 模型性能对比分析
- 混淆矩阵导出
- 推理可视化对比

**支持的模型：**
| 模型 | 输入尺寸 | 适用场景 |
|------|---------|---------|
| YOLOv8n | 640×640 | 边缘设备部署 |
| YOLOv8s/m/l | 640×640 | 平衡精度速度 |
| YOLOv11-L | 1280×1280 | 高精度需求 |
| YOLOv12 | 1280×1280 | 最佳精度 |

## 📊 评估指标

训练完成后，系统将自动生成以下评估报告：

- **性能对比表**: mAP@50, mAP@50-95, Precision, Recall
- **训练曲线**: 损失下降、指标提升可视化
- **混淆矩阵**: 各类别检测准确性
- **推理对比**: 不同模型的检测结果对比

报告位置: `results/evaluation/comparison_report.md`

## 🎯 优化建议

### 针对水面反光
- 使用反光模拟增强 (`simulate_glare`)
- 应用CLAHE直方图均衡化
- 考虑偏振镜效果模拟

### 针对小目标检测
- 使用高分辨率输入 (1280×1280)
- 启用P2检测头 (4×4像素级)
- 使用多尺度训练 (MST)

详细优化指南: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)

## 🌿 Git工作流

本项目采用Git Flow分支策略：

```bash
# 功能开发
git checkout -b feat/data-pipeline
# ...开发...
git commit -m "feat(data): implement data cleaning"
git push origin feat/data-pipeline

# 合并到develop
git checkout develop
git merge feat/data-pipeline

# 发布版本
git checkout -b release/v1.0.0
git checkout main
git merge release/v1.0.0
git tag v1.0.0
```

详细指南: [GIT_WORKFLOW.md](GIT_WORKFLOW.md)

## 🔌 模型部署

### TensorRT导出

```python
from ultralytics import YOLO

model = YOLO('results/models/yolov8n_baseline/weights/best.pt')
model.export(format='engine', imgsz=640, half=True)
```

### ONNX Runtime推理

```python
import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession('best.onnx')
image = cv2.imread('test.jpg')
image = cv2.resize(image, (640, 640))
image = image.transpose(2, 0, 1) / 255.0
image = np.expand_dims(image, axis=0).astype(np.float32)

outputs = session.run(None, {'images': image})
```

## 📝 版本历史

- **v1.0.0** (2026-02-27)
  - 初始版本发布
  - 支持YOLOv8n和YOLOv12双模型对比
  - 实现水面场景专用数据增强
  - 完成数据集处理流水线

## 🤝 贡献指南

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feat/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送分支 (`git push origin feat/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 项目维护者: CV Engineer
- 邮箱: yueyezhifu@163.com
- 问题反馈: [Issues](https://github.com/your-org/anchor-detection/issues)

---

*本项目为水面光伏区智能运维解决方案的一部分*
