"""
水面光伏锚固系统模型训练与对比评估
支持YOLOv8n (Baseline) 和 YOLOv12 (SOTA)
Author: CV Engineer
Date: 2026-02-27
"""

import os
import cv2
import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.utils.callbacks import Callbacks
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    logger.error("Ultralytics库未安装，请运行: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False


@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str                          # 模型名称
    model_type: str                    # YOLO版本 (yolov8n, yolov12, etc.)
    pretrained: str                    # 预训练权重路径
    imgsz: int                         # 输入尺寸
    epochs: int                        # 训练轮数
    batch: int                         # 批次大小
    device: str                        # 训练设备
    workers: int                       # 数据加载线程
    optimizer: str                     # 优化器 (SGD/Adam/AdamW)
    lr0: float                         # 初始学习率
    lrf: float                         # 最终学习率系数
    momentum: float                    # 动量
    weight_decay: float                # 权重衰减
    warmup_epochs: float               # 预热轮数
    warmup_momentum: float             # 预热动量
    box: float                         # 边框损失权重
    cls: float                        # 分类损失权重
    dfl: float                        # 分布焦点损失权重
    patience: int                      # 早停耐心值
    save_period: int                   # 保存周期
    amp: bool                          # 混合精度训练
    exist_ok: bool                     # 覆盖已有结果
    resume: bool                      # 断点续训
    

class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, config: ModelConfig, data_yaml: str, output_dir: str = 'results/models'):
        """
        Args:
            config: 模型配置
            data_yaml: 数据配置文件路径
            output_dir: 模型输出目录
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics库未安装")
        
        self.config = config
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练结果存储
        self.results = {
            'train_loss': [],
            'val_map50': [],
            'val_map5095': [],
            'val_precision': [],
            'val_recall': [],
            'epochs': [],
        }
        
        # 回调函数
        self.callbacks = Callbacks()
        self._setup_callbacks()
        
        # 初始化模型
        self.model = None
        self.trainer = None
        
        logger.info(f"初始化训练器: {config.name}")
    
    def _setup_callbacks(self):
        """设置训练回调函数"""
        
        def on_train_epoch_end(trainer):
            """每轮训练结束回调"""
            epoch = trainer.epoch
            
            # 记录训练指标
            self.results['epochs'].append(epoch)
            
            # 提取损失
            if hasattr(trainer, 'loss_items'):
                self.results['train_loss'].append(float(trainer.loss_items[0]))
            
            # 验证指标 (每几轮记录一次)
            if hasattr(trainer, 'metrics') and trainer.metrics:
                metrics = trainer.metrics
                self.results['val_map50'].append(metrics.get('metrics/mAP50', 0))
                self.results['val_map5095'].append(metrics.get('metrics/mAP50-95', 0))
                self.results['val_precision'].append(metrics.get('metrics/precision', 0))
                self.results['val_recall'].append(metrics.get('metrics/recall', 0))
            
            logger.info(f"Epoch {epoch}/{self.config.epochs} 完成")
        
        def on_fit_epoch_end(trainer):
            """每轮拟合结束回调"""
            pass
        
        def on_train_end(trainer):
            """训练结束回调"""
            logger.info(f"训练完成: {self.config.name}")
            self._save_training_history()
        
        # 注册回调
        self.callbacks.register_action('on_train_epoch_end', on_train_epoch_end)
        self.callbacks.register_action('on_fit_epoch_end', on_fit_epoch_end)
        self.callbacks.register_action('on_train_end', on_train_end)
    
    def _save_training_history(self):
        """保存训练历史"""
        history_file = self.output_dir / f'{self.config.name}_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"训练历史保存至: {history_file}")
    
    def train(self) -> Dict:
        """
        执行模型训练
        
        Returns:
            训练结果字典
        """
        logger.info(f"开始训练 {self.config.name}...")
        
        # 加载预训练模型
        try:
            # 尝试加载指定版本的YOLO
            self.model = YOLO(self.config.pretrained)
            logger.info(f"加载模型: {self.config.pretrained}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("尝试使用YOLOv8n作为回退...")
            self.model = YOLO('yolov8n.pt')
        
        # 训练参数
        train_args = {
            'data': self.data_yaml,
            'imgsz': self.config.imgsz,
            'epochs': self.config.epochs,
            'batch': self.config.batch,
            'device': self.config.device,
            'workers': self.config.workers,
            'optimizer': self.config.optimizer,
            'lr0': self.config.lr0,
            'lrf': self.config.lrf,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'box': self.config.box,
            'cls': self.config.cls,
            'dfl': self.config.dfl,
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'project': str(self.output_dir),
            'name': self.config.name,
            'exist_ok': self.config.exist_ok,
            'amp': self.config.amp,
            'resume': self.config.resume,
            'verbose': True,
        }
        
        # 执行训练
        start_time = datetime.now()
        
        try:
            self.trainer = self.model.train(**train_args)
            
            # 获取最终指标
            final_metrics = {
                'mAP50': float(self.trainer.results_dict.get('metrics/mAP50', 0)),
                'mAP50-95': float(self.trainer.results_dict.get('metrics/mAP50-95', 0)),
                'precision': float(self.trainer.results_dict.get('metrics/precision', 0)),
                'recall': float(self.trainer.results_dict.get('metrics/recall', 0)),
            }
            
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            raise
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # 保存训练摘要
        summary = {
            'model_name': self.config.name,
            'config': asdict(self.config),
            'final_metrics': final_metrics,
            'training_duration': training_duration,
            'best_weights': str(self.output_dir / self.config.name / 'weights' / 'best.pt'),
        }
        
        summary_file = self.output_dir / f'{self.config.name}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"训练摘要保存至: {summary_file}")
        logger.info(f"训练耗时: {training_duration/3600:.2f} 小时")
        
        return summary
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        导出模型
        
        Args:
            format: 导出格式 (onnx, engine, tflite, etc.)
            
        Returns:
            导出文件路径
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        logger.info(f"导出模型为 {format} 格式...")
        
        try:
            export_path = self.model.export(format=format, imgsz=self.config.imgsz)
            logger.info(f"模型导出至: {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return ""


class ModelComparator:
    """模型对比评估器"""
    
    def __init__(self, results_dir: str = 'results/evaluation'):
        """
        Args:
            results_dir: 评估结果输出目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.summaries = []
    
    def add_model_result(self, summary: Dict):
        """添加模型训练结果"""
        self.summaries.append(summary)
    
    def generate_comparison_report(self) -> str:
        """
        生成对比评估报告
        
        Returns:
            报告文件路径
        """
        logger.info("生成模型对比报告...")
        
        if len(self.summaries) < 2:
            logger.warning("至少需要2个模型结果才能生成对比报告")
            return ""
        
        # 创建对比表格
        comparison_data = []
        for summary in self.summaries:
            metrics = summary['final_metrics']
            comparison_data.append({
                'Model': summary['model_name'],
                'mAP@50': f"{metrics['mAP50']:.4f}",
                'mAP@50-95': f"{metrics['mAP50-95']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Duration(h)': f"{summary['training_duration']/3600:.2f}",
                'Image Size': summary['config']['imgsz'],
                'Epochs': summary['config']['epochs'],
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 保存CSV
        csv_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 生成可视化图表
        self._plot_comparison_charts()
        
        # 生成Markdown报告
        report_path = self.results_dir / 'comparison_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 水面光伏锚固系统检测模型对比评估报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 模型配置对比\n\n")
            f.write("| 模型 | 输入尺寸 | Epochs | Batch | 优化器 | 学习率 |\n")
            f.write("|------|----------|--------|-------|--------|--------|\n")
            for summary in self.summaries:
                cfg = summary['config']
                f.write(f"| {summary['model_name']} | {cfg['imgsz']} | {cfg['epochs']} | "
                       f"{cfg['batch']} | {cfg['optimizer']} | {cfg['lr0']} |\n")
            
            f.write("\n## 2. 性能指标对比\n\n")
            f.write(df.to_markdown(index=False))
            
            f.write("\n\n## 3. 关键发现\n\n")
            
            # 找出最佳模型
            best_map50 = max(self.summaries, key=lambda x: x['final_metrics']['mAP50'])
            best_map5095 = max(self.summaries, key=lambda x: x['final_metrics']['mAP50-95'])
            
            f.write(f"- **最佳 mAP@50**: {best_map50['model_name']} ({best_map50['final_metrics']['mAP50']:.4f})\n")
            f.write(f"- **最佳 mAP@50-95**: {best_map5095['model_name']} ({best_map5095['final_metrics']['mAP50-95']:.4f})\n")
            
            # 分析差异
            map50_diff = best_map50['final_metrics']['mAP50'] - min(
                s['final_metrics']['mAP50'] for s in self.summaries
            )
            f.write(f"- **mAP@50 提升**: {map50_diff*100:.2f}%\n")
            
            f.write("\n## 4. 结论与建议\n\n")
            f.write("根据对比实验结果:\n\n")
            f.write(f"1. **高精度场景推荐**: 使用 {best_map5095['model_name']}，其在COCO指标上表现最优。\n")
            f.write(f"2. **边缘部署推荐**: 考虑使用轻量级模型如YOLOv8n，权衡精度与速度。\n")
            f.write("3. **进一步优化方向**: \n")
            f.write("   - 针对水面反光场景增加数据增强\n")
            f.write("   - 使用更高分辨率输入(1280+)提升小目标检测\n")
            f.write("   - 考虑模型集成策略\n")
            
            f.write("\n## 5. 可视化图表\n\n")
            f.write("对比图表保存位置:\n")
            f.write(f"- 指标对比图: {self.results_dir / 'metrics_comparison.png'}\n")
            f.write(f"- 训练曲线对比: {self.results_dir / 'training_curves.png'}\n")
        
        logger.info(f"对比报告生成完成: {report_path}")
        return str(report_path)
    
    def _plot_comparison_charts(self):
        """绘制对比图表"""
        # 指标对比柱状图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = ['mAP50', 'mAP50-95', 'precision', 'recall']
        titles = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
        
        for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
            ax = axes[idx // 2, idx % 2]
            
            model_names = [s['model_name'] for s in self.summaries]
            values = [s['final_metrics'][metric] for s in self.summaries]
            
            bars = ax.bar(model_names, values, color=['#3498db', '#e74c3c'][:len(model_names)])
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 训练曲线对比 (如果有历史数据)
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """绘制训练曲线对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
        
        for summary in self.summaries:
            history_path = Path('results/models') / f"{summary['model_name']}_history.json"
            if not history_path.exists():
                continue
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            epochs = history.get('epochs', [])
            
            # mAP50曲线
            if history.get('val_map50'):
                axes[0, 0].plot(epochs, history['val_map50'], 
                               label=summary['model_name'], linewidth=2)
            
            # mAP50-95曲线
            if history.get('val_map5095'):
                axes[0, 1].plot(epochs, history['val_map5095'], 
                               label=summary['model_name'], linewidth=2)
            
            # Precision曲线
            if history.get('val_precision'):
                axes[1, 0].plot(epochs, history['val_precision'], 
                               label=summary['model_name'], linewidth=2)
            
            # Recall曲线
            if history.get('val_recall'):
                axes[1, 1].plot(epochs, history['val_recall'], 
                               label=summary['model_name'], linewidth=2)
        
        titles = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
        for idx, title in enumerate(titles):
            ax = axes[idx // 2, idx % 2]
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class InferenceVisualizer:
    """推理可视化器"""
    
    def __init__(self, results_dir: str = 'results/evaluation'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_inference(self, 
                         model_paths: Dict[str, str],
                         test_images: List[str],
                         conf_threshold: float = 0.25) -> str:
        """
        对比不同模型的推理结果
        
        Args:
            model_paths: 模型路径字典 {model_name: path}
            test_images: 测试图像路径列表
            conf_threshold: 置信度阈值
            
        Returns:
            输出目录路径
        """
        logger.info("生成推理可视化对比...")
        
        output_dir = self.results_dir / 'inference_comparison'
        output_dir.mkdir(exist_ok=True)
        
        # 加载所有模型
        models = {}
        for name, path in model_paths.items():
            try:
                models[name] = YOLO(path)
                logger.info(f"加载模型: {name} from {path}")
            except Exception as e:
                logger.error(f"加载模型失败 {name}: {e}")
        
        # 对每张测试图像进行推理
        for img_path in test_images:
            img_name = Path(img_path).stem
            
            # 创建对比图
            fig, axes = plt.subplots(1, len(models) + 1, figsize=(5 * (len(models) + 1), 6))
            fig.suptitle(f'Inference Comparison: {img_name}', fontsize=14)
            
            # 显示原图
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(original_img)
            axes[0].set_title('Original', fontsize=12)
            axes[0].axis('off')
            
            # 各模型推理结果
            for idx, (model_name, model) in enumerate(models.items(), 1):
                results = model(img_path, conf=conf_threshold)
                
                # 绘制结果
                annotated_frame = results[0].plot()
                axes[idx].imshow(annotated_frame)
                axes[idx].set_title(f'{model_name}', fontsize=12)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'compare_{img_name}.png', dpi=200, bbox_inches='tight')
            plt.close()
        
        logger.info(f"推理可视化保存至: {output_dir}")
        return str(output_dir)
    
    def visualize_defect_cases(self,
                              model_path: str,
                              test_images: List[str],
                              defect_types: List[str],
                              output_subdir: str = 'defect_cases') -> str:
        """
        可视化典型缺陷案例
        
        Args:
            model_path: 模型路径
            test_images: 测试图像路径列表
            defect_types: 关注的缺陷类型列表
            output_subdir: 输出子目录
            
        Returns:
            输出目录路径
        """
        logger.info("可视化典型缺陷案例...")
        
        output_dir = self.results_dir / output_subdir
        output_dir.mkdir(exist_ok=True)
        
        model = YOLO(model_path)
        
        for img_path in test_images:
            results = model(img_path, conf=0.25)
            
            # 检查是否包含目标缺陷类型
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
            
            if any(defect in ' '.join(detected_classes) for defect in defect_types):
                annotated = results[0].plot()
                
                img_name = Path(img_path).stem
                cv2.imwrite(str(output_dir / f'defect_{img_name}.jpg'), annotated)
        
        logger.info(f"缺陷案例可视化保存至: {output_dir}")
        return str(output_dir)


def create_yolov8n_config() -> ModelConfig:
    """创建YOLOv8n配置 (Baseline轻量级)"""
    return ModelConfig(
        name='yolov8n_baseline',
        model_type='yolov8n',
        pretrained='yolov8n.pt',
        imgsz=640,
        epochs=100,
        batch=16,
        device='auto',
        workers=8,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=50,
        save_period=10,
        amp=True,
        exist_ok=True,
        resume=False,
    )


def create_yolov12_config() -> ModelConfig:
    """创建YOLOv12配置 (SOTA高精度)"""
    # 注意：YOLOv12在2026年可能已发布，如果不可用则使用yolov11l作为替代
    try:
        # 尝试检测YOLOv12是否可用
        test_model = YOLO('yolov12l.pt')
        pretrained = 'yolov12l.pt'
        model_type = 'yolov12l'
    except:
        logger.warning("YOLOv12不可用，使用YOLOv11-Large作为替代")
        pretrained = 'yolov11l.pt'
        model_type = 'yolov11l'
    
    return ModelConfig(
        name='yolov12_sota' if model_type.startswith('yolov12') else 'yolov11l_sota',
        model_type=model_type,
        pretrained=pretrained,
        imgsz=1280,  # 利用高分辨率优势
        epochs=150,
        batch=8,     # 大分辨率减小batch
        device='auto',
        workers=8,
        optimizer='AdamW',  # 更好的收敛性
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.05,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=75,
        save_period=10,
        amp=True,  # 混合精度
        exist_ok=True,
        resume=False,
    )


def main():
    """主入口 - 执行双模型训练与对比"""
    
    # 数据配置路径
    DATA_YAML = 'data/processed/data.yaml'
    
    if not os.path.exists(DATA_YAML):
        logger.error(f"数据配置文件不存在: {DATA_YAML}")
        logger.info("请先运行 data_prep/dataset_processor.py 准备数据集")
        return
    
    # 初始化对比器
    comparator = ModelComparator()
    
    # ============ 模型1: YOLOv8n (Baseline) ============
    logger.info("=" * 60)
    logger.info("开始训练 Baseline 模型: YOLOv8n")
    logger.info("=" * 60)
    
    config_v8n = create_yolov8n_config()
    trainer_v8n = YOLOTrainer(config_v8n, DATA_YAML)
    
    try:
        summary_v8n = trainer_v8n.train()
        comparator.add_model_result(summary_v8n)
        
        # 导出ONNX (边缘部署)
        trainer_v8n.export_model('onnx')
        
    except Exception as e:
        logger.error(f"YOLOv8n训练失败: {e}")
    
    # ============ 模型2: YOLOv12 (SOTA) ============
    logger.info("=" * 60)
    logger.info("开始训练 SOTA 模型: YOLOv12")
    logger.info("=" * 60)
    
    config_v12 = create_yolov12_config()
    trainer_v12 = YOLOTrainer(config_v12, DATA_YAML)
    
    try:
        summary_v12 = trainer_v12.train()
        comparator.add_model_result(summary_v12)
        
        # 导出ONNX
        trainer_v12.export_model('onnx')
        
    except Exception as e:
        logger.error(f"YOLOv12训练失败: {e}")
    
    # ============ 生成对比报告 ============
    logger.info("=" * 60)
    logger.info("生成对比评估报告")
    logger.info("=" * 60)
    
    comparator.generate_comparison_report()
    
    # ============ 推理可视化对比 ============
    logger.info("=" * 60)
    logger.info("生成推理可视化")
    logger.info("=" * 60)
    
    visualizer = InferenceVisualizer()
    
    # 获取测试图像
    test_images = list(Path('data/processed/images/test').glob('*.jpg'))[:10]
    
    if len(test_images) > 0:
        model_paths = {
            'YOLOv8n': str(Path('results/models') / 'yolov8n_baseline' / 'weights' / 'best.pt'),
            'YOLOv12': str(Path('results/models') / 'yolov12_sota' / 'weights' / 'best.pt'),
        }
        
        # 只使用存在的模型
        model_paths = {k: v for k, v in model_paths.items() if os.path.exists(v)}
        
        if model_paths:
            test_image_paths = [str(p) for p in test_images]
            visualizer.compare_inference(model_paths, test_image_paths)
            
            # 缺陷案例可视化 (使用YOLOv12)
            if 'YOLOv12' in model_paths:
                visualizer.visualize_defect_cases(
                    model_paths['YOLOv12'],
                    test_image_paths,
                    defect_types=['loose', 'missing', 'damaged']
                )
    
    logger.info("=" * 60)
    logger.info("所有任务完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
