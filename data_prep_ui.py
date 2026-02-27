"""
水面光伏锚固系统 - 数据预处理模块 UI
Dataset Preparation Web Interface

使用方式:
    python data_prep_ui.py
    
依赖:
    pip install gradio

Author: CV Engineer
Date: 2026-02-27
"""

import os
import sys
import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

import gradio as gr

# 导入核心处理模块
from data_prep.dataset_processor import DatasetProcessor, ClassMapper

# 全局变量存储处理状态
processing_status = {
    'is_running': False,
    'progress': 0,
    'message': '',
    'log': []
}


def add_log(message: str):
    """添加日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    processing_status['log'].append(log_entry)
    if len(processing_status['log']) > 100:
        processing_status['log'] = processing_status['log'][-100:]
    return '\n'.join(processing_status['log'])


def process_dataset(
    raw_data_dir: str,
    output_dir: str,
    min_resolution_w: int,
    min_resolution_h: int,
    min_target_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_augmentation: bool,
    enable_ripple: bool,
    enable_glare: bool,
    enable_algae: bool,
    visualize_num: int
) -> str:
    """
    执行数据集处理
    
    Returns:
        处理结果消息
    """
    global processing_status
    
    if processing_status['is_running']:
        return "错误: 已有任务在运行中"
    
    # 验证路径
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        return f"错误: 原始数据目录不存在: {raw_data_dir}"
    
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        return f"错误: 数据集划分比例之和必须等于1.0 (当前: {total_ratio})"
    
    processing_status['is_running'] = True
    processing_status['progress'] = 0
    processing_status['log'] = []
    
    try:
        add_log("=" * 50)
        add_log("开始数据预处理")
        add_log(f"原始数据: {raw_data_dir}")
        add_log(f"输出目录: {output_dir}")
        add_log("=" * 50)
        
        # 创建处理器
        processor = DatasetProcessor(
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            min_resolution=(min_resolution_w, min_resolution_h),
            min_target_size=min_target_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_segmentation=False
        )
        
        add_log("正在加载数据集...")
        
        # 执行处理
        processor.process_dataset()
        
        processing_status['progress'] = 80
        add_log("数据划分完成")
        
        # 可视化
        if visualize_num > 0:
            add_log(f"正在生成 {visualize_num} 张可视化样本...")
            processor.visualize_dataset(num_samples=visualize_num)
            add_log("可视化完成")
        
        processing_status['progress'] = 100
        processing_status['is_running'] = False
        
        add_log("=" * 50)
        add_log("处理完成!")
        add_log(f"输出位置: {output_dir}")
        add_log("=" * 50)
        
        return "✅ 处理成功完成!\n\n" + '\n'.join(processing_status['log'][-20:])
        
    except Exception as e:
        processing_status['is_running'] = False
        add_log(f"❌ 错误: {str(e)}")
        return f"处理失败: {str(e)}\n\n日志:\n" + '\n'.join(processing_status['log'][-20:])


def get_class_info() -> str:
    """获取类别信息"""
    info = "## 16类检测目标\n\n"
    info += "| ID | 类别名称 | 说明 |\n"
    info += "|----|---------|------|\n"
    
    for idx, name in enumerate(ClassMapper.CLASS_NAMES):
        component = name.rsplit('_', 1)[0].replace('_', ' ').title()
        state = name.rsplit('_', 1)[1]
        info += f"| {idx} | {name} | {component} - {state} |\n"
    
    return info


def create_ui():
    """创建Gradio UI界面"""
    
    with gr.Blocks(title="水面光伏锚固系统 - 数据预处理", theme=gr.themes.Soft()) as demo:
        
        # 标题
        gr.Markdown("""
        # 🌊 水面光伏锚固系统智能检测 - 数据预处理
        
        本工具用于准备训练数据集，包括数据清洗、格式转换、数据增强和数据集划分。
        """)
        
        with gr.Tabs():
            
            # ========== Tab 1: 数据处理 ==========
            with gr.TabItem("📁 数据处理"):
                
                with gr.Row():
                    # 左侧：路径配置
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 路径配置")
                        
                        raw_dir_input = gr.Textbox(
                            label="原始数据目录",
                            placeholder="data/raw",
                            value="data/raw",
                            info="包含图像和标注文件的目录"
                        )
                        
                        output_dir_input = gr.Textbox(
                            label="输出目录",
                            placeholder="data/processed",
                            value="data/processed",
                            info="处理后数据的保存位置"
                        )
                        
                        gr.Markdown("### 📊 数据集划分")
                        
                        train_ratio = gr.Slider(
                            minimum=0.5, maximum=0.9, value=0.7, step=0.05,
                            label="训练集比例"
                        )
                        val_ratio = gr.Slider(
                            minimum=0.1, maximum=0.3, value=0.2, step=0.05,
                            label="验证集比例"
                        )
                        test_ratio = gr.Slider(
                            minimum=0.0, maximum=0.2, value=0.1, step=0.05,
                            label="测试集比例"
                        )
                    
                    # 右侧：参数配置
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 处理参数")
                        
                        with gr.Row():
                            min_res_w = gr.Number(
                                label="最小宽度",
                                value=1920,
                                minimum=640,
                                step=10
                            )
                            min_res_h = gr.Number(
                                label="最小高度",
                                value=1080,
                                minimum=480,
                                step=10
                            )
                        
                        min_target_size = gr.Slider(
                            minimum=30, maximum=100, value=50, step=5,
                            label="最小目标尺寸 (像素)",
                            info="小于此尺寸的目标将被过滤"
                        )
                        
                        gr.Markdown("### 🎨 数据增强")
                        
                        use_aug = gr.Checkbox(
                            label="启用数据增强",
                            value=True
                        )
                        
                        with gr.Row():
                            enable_ripple = gr.Checkbox(
                                label="水面波纹模拟",
                                value=True
                            )
                            enable_glare = gr.Checkbox(
                                label="阳光反光模拟",
                                value=True
                            )
                            enable_algae = gr.Checkbox(
                                label="藻类覆盖模拟",
                                value=True
                            )
                        
                        visualize_num = gr.Slider(
                            minimum=0, maximum=50, value=20, step=5,
                            label="可视化样本数量",
                            info="生成带标注框的预览图"
                        )
                
                # 处理按钮和输出
                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 开始处理",
                        variant="primary",
                        size="lg"
                    )
                
                output_log = gr.Textbox(
                    label="处理日志",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
                
                # 绑定按钮事件
                process_btn.click(
                    fn=process_dataset,
                    inputs=[
                        raw_dir_input, output_dir_input,
                        min_res_w, min_res_h, min_target_size,
                        train_ratio, val_ratio, test_ratio,
                        use_aug, enable_ripple, enable_glare, enable_algae,
                        visualize_num
                    ],
                    outputs=output_log
                )
            
            # ========== Tab 2: 类别信息 ==========
            with gr.TabItem("📋 类别说明"):
                gr.Markdown(get_class_info())
                
                gr.Markdown("""
                ### 🎨 颜色说明
                
                可视化时使用以下颜色编码：
                
                - 🔵 **蓝色系** - 锚绳 (Anchor Rope)
                - 🟠 **橙色系** - 连接件 (Anchor Connector)
                - 🟢 **绿色系** - 锚块 (Anchor Block)
                - 🟣 **紫色系** - 浮筒连接件 (Float Fastener)
                
                ### 📝 状态说明
                
                - **normal** - 正常状态，无缺陷
                - **loose** - 松动/松弛/移位
                - **missing** - 断裂/缺失/脱落
                - **damaged** - 破损/腐蚀/损坏
                """)
            
            # ========== Tab 3: 使用帮助 ==========
            with gr.TabItem("❓ 使用帮助"):
                gr.Markdown("""
                ### 📖 快速开始
                
                1. **准备原始数据**
                   - 将图像文件放入 `data/raw/` 目录
                   - 标注文件（JSON/XML）与图像同名
                
                2. **配置参数**
                   - 设置输入/输出路径
                   - 调整数据集划分比例
                   - 选择数据增强选项
                
                3. **开始处理**
                   - 点击"开始处理"按钮
                   - 等待处理完成
                
                ### 📁 输出结构
                
                ```
                data/processed/
                ├── images/
                │   ├── train/     # 训练集图像
                │   ├── val/       # 验证集图像
                │   └── test/      # 测试集图像
                ├── labels/
                │   ├── train/     # 训练集标注
                │   ├── val/       # 验证集标注
                │   └── test/      # 测试集标注
                └── data.yaml      # 数据集配置
                ```
                
                ### ⚠️ 注意事项
                
                - 确保磁盘有足够空间（原始数据的2-3倍）
                - 处理过程中请勿关闭窗口
                - 建议先小批量测试再大批量处理
                """)
        
        # 底部信息
        gr.Markdown("---")
        gr.Markdown("水面光伏锚固系统智能检测 | 数据预处理模块 v1.0")
    
    return demo


def main():
    """主入口"""
    print("=" * 60)
    print("启动数据预处理 UI")
    print("=" * 60)
    print("请在浏览器中访问: http://127.0.0.1:7860")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
