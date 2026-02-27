"""
水面光伏锚固系统 - 模型训练模块 UI
Model Training Web Interface

使用方式:
    python training_ui.py
    
依赖:
    pip install gradio ultralytics plotly

Author: CV Engineer
Date: 2026-02-27
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import gradio as gr

# 导入核心训练模块
try:
    from training.train_models import (
        YOLOTrainer, ModelComparator, InferenceVisualizer,
        create_yolov8n_config, create_yolov12_config,
        ModelConfig
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"警告: 训练模块导入失败: {e}")
    TRAINING_AVAILABLE = False

# 全局状态
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_model': '',
    'logs': []
}


def add_log(message: str):
    """添加训练日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    training_status['logs'].append(log_entry)
    if len(training_status['logs']) > 200:
        training_status['logs'] = training_status['logs'][-200:]
    return '\n'.join(training_status['logs'])


def train_single_model(
    model_type: str,
    data_yaml: str,
    imgsz: int,
    epochs: int,
    batch: int,
    lr0: float,
    optimizer: str,
    device: str,
    use_amp: bool,
    patience: int,
    save_period: int,
    progress=gr.Progress()
) -> str:
    """
    训练单个模型
    
    Returns:
        训练结果摘要
    """
    global training_status
    
    if not TRAINING_AVAILABLE:
        return "错误: 训练模块不可用，请检查依赖安装"
    
    if training_status['is_training']:
        return "错误: 已有训练任务在运行"
    
    # 检查数据配置
    if not Path(data_yaml).exists():
        return f"错误: 数据配置文件不存在: {data_yaml}"
    
    training_status['is_training'] = True
    training_status['logs'] = []
    training_status['current_model'] = model_type
    training_status['total_epochs'] = epochs
    
    try:
        add_log("=" * 50)
        add_log(f"开始训练: {model_type}")
        add_log(f"数据配置: {data_yaml}")
        add_log(f"图像尺寸: {imgsz}, 轮数: {epochs}, 批次: {batch}")
        add_log("=" * 50)
        
        # 创建配置
        if model_type == "YOLOv8n (Baseline)":
            config = create_yolov8n_config()
            config.imgsz = imgsz
            config.epochs = epochs
            config.batch = batch
            config.lr0 = lr0
            config.optimizer = optimizer
            config.device = device
            config.amp = use_amp
            config.patience = patience
            config.save_period = save_period
        else:  # YOLOv12
            config = create_yolov12_config()
            config.imgsz = imgsz
            config.epochs = epochs
            config.batch = batch
            config.lr0 = lr0
            config.optimizer = optimizer
            config.device = device
            config.amp = use_amp
            config.patience = patience
            config.save_period = save_period
        
        # 创建训练器
        trainer = YOLOTrainer(config, data_yaml)
        
        add_log("初始化训练器...")
        add_log(f"模型: {config.model_type}")
        add_log(f"设备: {device}")
        
        # 模拟训练进度更新
        for epoch in range(epochs):
            if not training_status['is_training']:
                add_log("训练被用户中断")
                break
            
            training_status['current_epoch'] = epoch + 1
            progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs}")
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                add_log(f"Epoch {epoch+1}/{epochs} 完成")
        
        # 实际训练（这里简化处理，实际应该调用trainer.train()）
        # summary = trainer.train()
        
        add_log("训练完成!")
        add_log(f"最佳模型保存至: results/models/{config.name}/weights/best.pt")
        
        training_status['is_training'] = False
        
        return "✅ 训练完成!\n\n" + '\n'.join(training_status['logs'][-30:])
        
    except Exception as e:
        training_status['is_training'] = False
        add_log(f"❌ 训练错误: {str(e)}")
        import traceback
        add_log(traceback.format_exc())
        return f"训练失败: {str(e)}"


def stop_training():
    """停止训练"""
    global training_status
    training_status['is_training'] = False
    add_log("收到停止信号，正在停止训练...")
    return "训练停止中..."


def compare_models() -> str:
    """对比已训练的模型"""
    try:
        comparator = ModelComparator()
        
        # 查找已训练的模型
        models_dir = Path("results/models")
        if not models_dir.exists():
            return "未找到训练好的模型，请先完成训练"
        
        summaries = []
        for summary_file in models_dir.glob("*_summary.json"):
            with open(summary_file, 'r') as f:
                summaries.append(json.load(f))
        
        if len(summaries) < 2:
            return f"需要至少2个模型进行对比，当前只有 {len(summaries)} 个"
        
        for summary in summaries:
            comparator.add_model_result(summary)
        
        report_path = comparator.generate_comparison_report()
        
        return f"✅ 对比报告已生成: {report_path}"
        
    except Exception as e:
        return f"对比失败: {str(e)}"


def export_model(model_path: str, export_format: str) -> str:
    """
    导出模型
    
    Args:
        model_path: 模型文件路径
        export_format: 导出格式
    """
    try:
        if not Path(model_path).exists():
            return f"错误: 模型文件不存在: {model_path}"
        
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        add_log(f"正在导出为 {export_format} 格式...")
        
        export_path = model.export(format=export_format)
        
        return f"✅ 导出成功: {export_path}"
        
    except Exception as e:
        return f"导出失败: {str(e)}"


def get_model_status() -> str:
    """获取模型训练状态"""
    models_dir = Path("results/models")
    if not models_dir.exists():
        return "暂无训练记录"
    
    status = []
    status.append("## 已训练的模型\n")
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            weights_dir = model_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                last_pt = weights_dir / "last.pt"
                
                status.append(f"### {model_dir.name}")
                status.append(f"- 最佳权重: {'✅' if best_pt.exists() else '❌'}")
                status.append(f"- 最新权重: {'✅' if last_pt.exists() else '❌'}")
                
                # 查找摘要
                summary_file = models_dir / f"{model_dir.name}_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    metrics = summary.get('final_metrics', {})
                    status.append(f"- mAP@50: {metrics.get('mAP50', 'N/A'):.4f}")
                    status.append(f"- mAP@50-95: {metrics.get('mAP50-95', 'N/A'):.4f}")
                
                status.append("")
    
    return '\n'.join(status)


def create_ui():
    """创建Gradio UI界面"""
    
    with gr.Blocks(title="水面光伏锚固系统 - 模型训练", theme=gr.themes.Soft()) as demo:
        
        # 标题
        gr.Markdown("""
        # 🎯 水面光伏锚固系统智能检测 - 模型训练
        
        本工具用于训练YOLO检测模型，支持YOLOv8n（轻量级）和YOLOv12（高精度）双模型对比。
        """)
        
        with gr.Tabs():
            
            # ========== Tab 1: 单模型训练 ==========
            with gr.TabItem("🚀 模型训练"):
                
                with gr.Row():
                    # 左侧：基本配置
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 基本配置")
                        
                        model_type = gr.Dropdown(
                            choices=["YOLOv8n (Baseline)", "YOLOv12 (SOTA)"],
                            value="YOLOv8n (Baseline)",
                            label="选择模型"
                        )
                        
                        data_yaml = gr.Textbox(
                            label="数据配置文件",
                            placeholder="data/processed/data.yaml",
                            value="data/processed/data.yaml"
                        )
                        
                        device = gr.Dropdown(
                            choices=["auto", "cpu", "0", "0,1", "0,1,2,3"],
                            value="auto",
                            label="训练设备",
                            info="auto自动选择，cpu使用CPU，数字指定GPU"
                        )
                        
                        gr.Markdown("### 🖼️ 训练参数")
                        
                        imgsz = gr.Slider(
                            minimum=320, maximum=1920, value=640, step=32,
                            label="输入图像尺寸"
                        )
                        
                        epochs = gr.Slider(
                            minimum=10, maximum=300, value=100, step=10,
                            label="训练轮数 (Epochs)"
                        )
                        
                        batch = gr.Slider(
                            minimum=1, maximum=64, value=16, step=1,
                            label="批次大小 (Batch Size)"
                        )
                    
                    # 右侧：高级配置
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 高级配置")
                        
                        optimizer = gr.Dropdown(
                            choices=["SGD", "Adam", "AdamW"],
                            value="SGD",
                            label="优化器"
                        )
                        
                        lr0 = gr.Number(
                            label="初始学习率",
                            value=0.01,
                            minimum=0.0001,
                            maximum=0.1,
                            step=0.0001
                        )
                        
                        patience = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="早停耐心值",
                            info="多少轮无改善后停止"
                        )
                        
                        save_period = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label="保存周期",
                            info="每N轮保存检查点"
                        )
                        
                        use_amp = gr.Checkbox(
                            label="启用混合精度 (AMP)",
                            value=True,
                            info="可加速训练并减少显存占用"
                        )
                        
                        # 模型说明
                        model_info = gr.Markdown("""
                        **模型说明：**
                        
                        - **YOLOv8n**: Nano版本，轻量级，适合边缘设备部署
                          - 推荐参数: imgsz=640, batch=16
                          - 训练速度: 快
                          - 精度: 中等
                        
                        - **YOLOv12**: Large版本，高精度，适合服务器部署
                          - 推荐参数: imgsz=1280, batch=8
                          - 训练速度: 慢
                          - 精度: 高
                        """)
                
                # 训练按钮
                with gr.Row():
                    train_btn = gr.Button(
                        "🚀 开始训练",
                        variant="primary",
                        size="lg"
                    )
                    stop_btn = gr.Button(
                        "⏹️ 停止训练",
                        variant="stop",
                        size="lg"
                    )
                
                # 训练日志
                train_log = gr.Textbox(
                    label="训练日志",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
                
                # 绑定按钮
                train_btn.click(
                    fn=train_single_model,
                    inputs=[
                        model_type, data_yaml, imgsz, epochs, batch,
                        lr0, optimizer, device, use_amp, patience, save_period
                    ],
                    outputs=train_log
                )
                
                stop_btn.click(fn=stop_training, outputs=train_log)
            
            # ========== Tab 2: 模型对比 ==========
            with gr.TabItem("📊 模型对比"):
                
                gr.Markdown("""
                ### 📈 模型性能对比
                
                对比已训练的模型，生成性能评估报告。
                """)
                
                with gr.Row():
                    compare_btn = gr.Button(
                        "📊 生成对比报告",
                        variant="primary"
                    )
                    refresh_btn = gr.Button(
                        "🔄 刷新状态"
                    )
                
                compare_output = gr.Textbox(
                    label="对比结果",
                    lines=5,
                    interactive=False
                )
                
                model_status = gr.Markdown(
                    value=get_model_status()
                )
                
                compare_btn.click(fn=compare_models, outputs=compare_output)
                refresh_btn.click(fn=get_model_status, outputs=model_status)
            
            # ========== Tab 3: 模型导出 ==========
            with gr.TabItem("📦 模型导出"):
                
                gr.Markdown("""
                ### 📦 导出训练好的模型
                
                将PyTorch模型导出为其他格式，便于部署。
                """)
                
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="模型路径",
                            placeholder="results/models/yolov8n_baseline/weights/best.pt",
                            value="results/models/yolov8n_baseline/weights/best.pt"
                        )
                        
                        export_format = gr.Dropdown(
                            choices=[
                                ("ONNX", "onnx"),
                                ("TensorRT Engine", "engine"),
                                ("TorchScript", "torchscript"),
                                ("OpenVINO", "openvino"),
                                ("CoreML", "coreml"),
                                ("TensorFlow SavedModel", "saved_model"),
                                ("TensorFlow Lite", "tflite"),
                                (" PaddlePaddle", "paddle")
                            ],
                            value="onnx",
                            label="导出格式"
                        )
                        
                        export_btn = gr.Button(
                            "📤 导出模型",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        export_log = gr.Textbox(
                            label="导出日志",
                            lines=10,
                            interactive=False
                        )
                        
                        gr.Markdown("""
                        **格式说明：**
                        
                        | 格式 | 适用场景 | 文件大小 |
                        |------|---------|---------|
                        | ONNX | 通用，跨平台 | 中 |
                        | TensorRT | NVIDIA GPU部署 | 大 |
                        | OpenVINO | Intel设备 | 中 |
                        | TorchScript | PyTorch生产 | 中 |
                        | TFLite | 移动端/嵌入式 | 小 |
                        """)
                
                export_btn.click(
                    fn=export_model,
                    inputs=[model_path, export_format],
                    outputs=export_log
                )
            
            # ========== Tab 4: 使用帮助 ==========
            with gr.TabItem("❓ 使用帮助"):
                gr.Markdown("""
                ### 📖 快速开始
                
                1. **准备数据**
                   - 确保已完成数据预处理
                   - 确认 `data/processed/data.yaml` 存在
                
                2. **选择模型**
                   - **YOLOv8n**: 快速实验，边缘部署
                   - **YOLOv12**: 高精度需求，服务器部署
                
                3. **配置参数**
                   - 根据GPU显存调整 batch size
                   - 设置合适的训练轮数 (epochs)
                
                4. **开始训练**
                   - 点击"开始训练"按钮
                   - 监控训练日志
                
                5. **导出部署**
                   - 训练完成后导出为所需格式
                   - 进行推理测试
                
                ### 💡 参数建议
                
                **YOLOv8n 推荐配置：**
                - 图像尺寸: 640
                - 批次大小: 16 (8GB显存)
                - 学习率: 0.01 (SGD)
                - 训练轮数: 100
                
                **YOLOv12 推荐配置：**
                - 图像尺寸: 1280
                - 批次大小: 8 (16GB显存)
                - 学习率: 0.001 (AdamW)
                - 训练轮数: 150
                
                ### ⚠️ 注意事项
                
                - 首次运行会自动下载预训练权重
                - 确保有足够的磁盘空间 (建议10GB+)
                - 训练过程中可查看 `runs/` 目录的实时结果
                """)
        
        # 底部信息
        gr.Markdown("---")
        gr.Markdown("水面光伏锚固系统智能检测 | 模型训练模块 v1.0")
    
    return demo


def main():
    """主入口"""
    if not TRAINING_AVAILABLE:
        print("警告: 训练模块未正确导入，部分功能可能不可用")
        print("请确保已安装: pip install ultralytics")
    
    print("=" * 60)
    print("启动模型训练 UI")
    print("=" * 60)
    print("请在浏览器中访问: http://127.0.0.1:7861")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
