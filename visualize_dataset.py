"""
水面光伏锚固系统数据集可视化分析工具
Data Visualization & Analysis Module for Floating PV Anchor Detection

功能：
1. 随机抽样网格展示（4x5网格，20张图）
2. 统计图表生成（类别分布、目标尺寸、纵横比）
3. 难例挖掘可视化（difficult样本）
4. HTML交互式报告生成

依赖安装:
pip install opencv-python matplotlib seaborn plotly pandas numpy Pillow

Author: CV Data Engineer
Date: 2026-02-27
"""

import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 可视化库
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns

# 可选：Plotly交互式图表
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly未安装，将跳过HTML交互式报告生成")

# ==================== 配置常量 ====================

# 16个类别的定义
CLASS_NAMES = [
    'anchor_rope_normal', 'anchor_rope_loose', 'anchor_rope_missing', 'anchor_rope_damaged',
    'anchor_connector_normal', 'anchor_connector_loose', 'anchor_connector_missing', 'anchor_connector_damaged',
    'anchor_block_normal', 'anchor_block_loose', 'anchor_block_missing', 'anchor_block_damaged',
    'float_fastener_normal', 'float_fastener_loose', 'float_fastener_missing', 'float_fastener_damaged'
]

# 部件类型到ID映射
COMPONENT_TYPES = {
    'anchor_rope': [0, 1, 2, 3],
    'anchor_connector': [4, 5, 6, 7],
    'anchor_block': [8, 9, 10, 11],
    'float_fastener': [12, 13, 14, 15]
}

# 状态类型
STATES = ['normal', 'loose', 'missing', 'damaged']

# 颜色方案：不同部件使用不同色系
COMPONENT_COLORS = {
    'anchor_rope': {
        'base': (41, 128, 185),      # 蓝色系
        'normal': (52, 152, 219),     # 亮蓝
        'loose': (155, 89, 182),      # 紫蓝
        'missing': (231, 76, 60),     # 红
        'damaged': (230, 126, 34),    # 橙
    },
    'anchor_connector': {
        'base': (230, 126, 34),       # 橙色系
        'normal': (241, 196, 15),     # 黄橙
        'loose': (243, 156, 18),      # 橙
        'missing': (231, 76, 60),     # 红
        'damaged': (192, 57, 43),     # 深红
    },
    'anchor_block': {
        'base': (39, 174, 96),        # 绿色系
        'normal': (46, 204, 113),     # 翠绿
        'loose': (241, 196, 15),      # 黄
        'missing': (231, 76, 60),     # 红
        'damaged': (211, 84, 0),      # 深橙
    },
    'float_fastener': {
        'base': (142, 68, 173),       # 紫色系
        'normal': (155, 89, 182),     # 浅紫
        'loose': (125, 60, 152),      # 紫
        'missing': (231, 76, 60),     # 红
        'damaged': (44, 62, 80),      # 深蓝灰
    }
}

# 状态对应的线型和填充样式（Matplotlib格式）
STATE_STYLES = {
    'normal': {'linestyle': '-', 'hatch': None, 'alpha': 1.0, 'linewidth': 2},
    'loose': {'linestyle': '--', 'hatch': None, 'alpha': 0.9, 'linewidth': 2.5},
    'missing': {'linestyle': ':', 'hatch': '///', 'alpha': 0.8, 'linewidth': 3},
    'damaged': {'linestyle': '-.', 'hatch': '\\\\\\', 'alpha': 0.85, 'linewidth': 2.5}
}

# 背景复杂度标签
BACKGROUND_TAGS = {
    'glare': {'color': (255, 255, 0), 'text': '强反光', 'alpha': 0.3},
    'algae': {'color': (0, 255, 0), 'text': '藻类覆盖', 'alpha': 0.3},
    'ripple': {'color': (0, 255, 255), 'text': '波纹干扰', 'alpha': 0.3},
    'occlusion': {'color': (255, 0, 255), 'text': '遮挡', 'alpha': 0.3},
}


@dataclass
class Annotation:
    """标注数据结构"""
    class_id: int
    class_name: str
    component: str  # 部件类型
    state: str      # 状态
    x_center: float
    y_center: float
    width: float
    height: float
    difficult: bool = False
    confidence: Optional[float] = None
    
    @property
    def x_min(self) -> float:
        return self.x_center - self.width / 2
    
    @property
    def y_min(self) -> float:
        return self.y_center - self.height / 2
    
    @property
    def x_max(self) -> float:
        return self.x_center + self.width / 2
    
    @property
    def y_max(self) -> float:
        return self.y_center + self.height / 2
    
    @property
    def aspect_ratio(self) -> float:
        """纵横比"""
        return self.width / self.height if self.height > 0 else 0


@dataclass
class ImageInfo:
    """图像信息结构"""
    filename: str
    filepath: str
    width: int
    height: int
    annotations: List[Annotation]
    tags: List[str] = None  # 背景复杂度标签
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DatasetVisualizer:
    """数据集可视化分析器"""
    
    def __init__(self, 
                 data_yaml_path: str,
                 images_dir: str,
                 labels_dir: str,
                 output_dir: str = 'outputs',
                 min_pixel_size: int = 50):
        """
        初始化可视化器
        
        Args:
            data_yaml_path: 数据配置文件路径
            images_dir: 图像目录
            labels_dir: 标注目录
            output_dir: 输出目录
            min_pixel_size: 最小目标像素尺寸
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.min_pixel_size = min_pixel_size
        
        # 创建输出目录
        self.vis_grid_dir = self.output_dir / 'vis_grid'
        self.stats_dir = self.output_dir / 'stats'
        self.hard_cases_dir = self.output_dir / 'hard_cases'
        self.reports_dir = self.output_dir / 'reports'
        
        for d in [self.vis_grid_dir, self.stats_dir, self.hard_cases_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.images_info = self._load_dataset()
        
        print(f"✅ 加载完成: 共 {len(self.images_info)} 张图像, "
              f"{sum(len(img.annotations) for img in self.images_info)} 个标注")
    
    def _parse_class_name(self, class_id: int) -> Tuple[str, str, str]:
        """解析类别ID为部件和状态"""
        class_name = CLASS_NAMES[class_id]
        parts = class_name.split('_')
        
        # 确定部件类型
        if 'anchor_rope' in class_name:
            component = 'anchor_rope'
        elif 'anchor_connector' in class_name:
            component = 'anchor_connector'
        elif 'anchor_block' in class_name:
            component = 'anchor_block'
        elif 'float_fastener' in class_name:
            component = 'float_fastener'
        else:
            component = 'unknown'
        
        # 确定状态
        state = parts[-1] if parts[-1] in STATES else 'unknown'
        
        return class_name, component, state
    
    def _load_dataset(self) -> List[ImageInfo]:
        """加载数据集"""
        images_info = []
        
        # 查找所有图像
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(self.images_dir.rglob(f'*{ext}'))
        
        for img_path in image_files:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 查找对应的标注文件
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            annotations = []
            tags = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 解析difficult标记（自定义格式）
                            difficult = False
                            confidence = None
                            if len(parts) > 5:
                                for i, p in enumerate(parts[5:]):
                                    if p == 'difficult':
                                        difficult = True
                                    elif p.startswith('conf='):
                                        confidence = float(p.split('=')[1])
                            
                            # 解析类别
                            class_name, component, state = self._parse_class_name(class_id)
                            
                            anno = Annotation(
                                class_id=class_id,
                                class_name=class_name,
                                component=component,
                                state=state,
                                x_center=x_center,
                                y_center=y_center,
                                width=width,
                                height=height,
                                difficult=difficult,
                                confidence=confidence
                            )
                            annotations.append(anno)
                            
                            # 自动检测背景复杂度（基于启发式规则）
                            if difficult:
                                tags.append('occlusion')
            
            # 启发式检测背景复杂度
            tags.extend(self._detect_background_complexity(img))
            tags = list(set(tags))  # 去重
            
            info = ImageInfo(
                filename=img_path.name,
                filepath=str(img_path),
                width=w,
                height=h,
                annotations=annotations,
                tags=tags
            )
            images_info.append(info)
        
        return images_info
    
    def _detect_background_complexity(self, img: np.ndarray) -> List[str]:
        """
        启发式检测背景复杂度
        
        Returns:
            标签列表 ['glare', 'algae', 'ripple', ...]
        """
        tags = []
        h, w = img.shape[:2]
        
        # 检测强反光（高亮区域占比）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > 240
        bright_ratio = np.sum(bright_mask) / (h * w)
        if bright_ratio > 0.05:  # 超过5%高亮区域
            tags.append('glare')
        
        # 检测藻类覆盖（绿色区域占比）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 绿色HSV范围
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (h * w)
        if green_ratio > 0.1:  # 超过10%绿色区域
            tags.append('algae')
        
        # 检测波纹（高频纹理）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 500:  # 高方差表示纹理复杂
            tags.append('ripple')
        
        return tags
    
    def _get_color_for_annotation(self, anno: Annotation) -> Tuple[int, int, int]:
        """获取标注框颜色 (BGR格式用于OpenCV)"""
        colors = COMPONENT_COLORS.get(anno.component, COMPONENT_COLORS['anchor_rope'])
        return colors.get(anno.state, colors['base'])
    
    def _draw_annotation(self, img: np.ndarray, anno: Annotation, 
                         draw_style: str = 'opencv') -> np.ndarray:
        """
        绘制单个标注
        
        Args:
            img: 输入图像
            anno: 标注信息
            draw_style: 'opencv' 或 'matplotlib'
        """
        h, w = img.shape[:2]
        
        # 计算像素坐标
        x1 = int(anno.x_min * w)
        y1 = int(anno.y_min * h)
        x2 = int(anno.x_max * w)
        y2 = int(anno.y_max * h)
        
        # 确保坐标有效
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        color = self._get_color_for_annotation(anno)
        
        if draw_style == 'opencv':
            # OpenCV绘制
            style = STATE_STYLES.get(anno.state, STATE_STYLES['normal'])
            thickness = style['linewidth']
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 对于缺失/破损，添加斜线填充效果（简化版）
            if anno.state in ['missing', 'damaged']:
                # 绘制对角线
                for offset in range(0, max(x2-x1, y2-y1), 10):
                    pt1 = (x1 + offset, y1)
                    pt2 = (x1, y1 + offset)
                    if pt2[0] < x2 and pt2[1] < y2:
                        cv2.line(img, pt1, pt2, color, 1)
            
            # 绘制标签
            label = f"{anno.component.split('_')[-1]}:{anno.state}"
            if anno.confidence:
                label += f" {anno.confidence:.2f}"
            
            # 标签背景
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # difficult标记
            if anno.difficult:
                cv2.putText(img, "[HARD]", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img
    
    def _add_background_overlay(self, img: np.ndarray, tags: List[str]) -> np.ndarray:
        """添加背景复杂度叠加层"""
        overlay = img.copy()
        h, w = img.shape[:2]
        
        y_offset = 30
        for tag in tags:
            if tag in BACKGROUND_TAGS:
                tag_info = BACKGROUND_TAGS[tag]
                color = tag_info['color']
                text = tag_info['text']
                
                # 绘制半透明背景
                cv2.rectangle(overlay, (10, y_offset - 25), (150, y_offset + 5), 
                             color, -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
                # 绘制文字
                cv2.putText(img, f"⚠ {text}", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                y_offset += 35
        
        return img
    
    def visualize_random_grid(self, num_samples: int = 20, grid_size: Tuple[int, int] = (4, 5)):
        """
        随机抽样网格展示
        
        Args:
            num_samples: 抽样数量
            grid_size: 网格行列数 (rows, cols)
        """
        print(f"📊 生成随机抽样网格 ({grid_size[0]}x{grid_size[1]})...")
        
        rows, cols = grid_size
        assert rows * cols >= num_samples, "网格容量不足"
        
        # 随机抽样
        samples = random.sample(self.images_info, min(num_samples, len(self.images_info)))
        
        # 创建大图
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        fig.suptitle('Dataset Random Sample Visualization', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (img_info, ax) in enumerate(zip(samples, axes.flat)):
            # 读取图像
            img = cv2.imread(img_info.filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 绘制标注
            for anno in img_info.annotations:
                img = self._draw_annotation(img, anno, 'opencv')
            
            # 添加背景叠加层
            img = self._add_background_overlay(img, img_info.tags)
            
            # 显示
            ax.imshow(img)
            ax.set_title(f"{img_info.filename[:20]}...\n"
                        f"Objs: {len(img_info.annotations)}", 
                        fontsize=8)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(samples), rows * cols):
            axes.flat[idx].axis('off')
        
        # 添加图例
        legend_elements = []
        for comp, colors in COMPONENT_COLORS.items():
            comp_name = comp.replace('_', ' ').title()
            legend_elements.append(
                mpatches.Patch(color=np.array(colors['base'])/255, 
                              label=comp_name)
            )
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=4, fontsize=10, title='Components')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存
        output_path = self.vis_grid_dir / f'random_grid_{num_samples}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 网格图已保存: {output_path}")
    
    def generate_statistics(self):
        """生成统计图表"""
        print("📈 生成统计图表...")
        
        # 收集统计数据
        all_annotations = []
        for img_info in self.images_info:
            for anno in img_info.annotations:
                all_annotations.append({
                    'class_id': anno.class_id,
                    'class_name': anno.class_name,
                    'component': anno.component,
                    'state': anno.state,
                    'width_px': anno.width * img_info.width,
                    'height_px': anno.height * img_info.height,
                    'aspect_ratio': anno.aspect_ratio,
                    'area_px': anno.width * anno.height * img_info.width * img_info.height,
                    'difficult': anno.difficult,
                    'image': img_info.filename
                })
        
        df = pd.DataFrame(all_annotations)
        
        # ========== 1. 类别分布直方图 ==========
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 按类别统计
        class_counts = df['class_name'].value_counts().sort_index()
        colors = [self._get_color_for_bar(name) for name in class_counts.index]
        
        ax1 = axes[0]
        bars = ax1.bar(range(len(class_counts)), class_counts.values, color=colors)
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels(class_counts.index, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Class Distribution (16 Classes)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=7)
        
        # 按部件类型统计
        ax2 = axes[1]
        comp_counts = df['component'].value_counts()
        comp_colors = [np.array(COMPONENT_COLORS[c]['base'])/255 for c in comp_counts.index]
        bars2 = ax2.bar(comp_counts.index, comp_counts.values, color=comp_colors)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Component Type Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.stats_dir / 'class_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # ========== 2. 目标尺寸散点图 ==========
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 标记合格/不合格
        df['meets_requirement'] = (df['width_px'] >= self.min_pixel_size) & \
                                   (df['height_px'] >= self.min_pixel_size)
        
        # 绘制散点图
        scatter = ax.scatter(df['width_px'], df['height_px'], 
                           c=df['meets_requirement'].map({True: 'green', False: 'red'}),
                           alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        
        # 添加阈值线
        ax.axvline(x=self.min_pixel_size, color='red', linestyle='--', 
                  linewidth=2, label=f'Min Width ({self.min_pixel_size}px)')
        ax.axhline(y=self.min_pixel_size, color='red', linestyle='--', 
                  linewidth=2, label=f'Min Height ({self.min_pixel_size}px)')
        
        # 不合格样本标注
        unqualified = df[~df['meets_requirement']]
        if len(unqualified) > 0:
            ax.scatter(unqualified['width_px'], unqualified['height_px'], 
                      c='red', s=100, marker='x', linewidths=2, 
                      label=f'Unqualified (n={len(unqualified)})')
        
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title(f'Object Size Distribution (Requirement: ≥{self.min_pixel_size}×{self.min_pixel_size}px)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        qualified_pct = df['meets_requirement'].mean() * 100
        stats_text = f"Total: {len(df)}\nQualified: {qualified_pct:.1f}%\n"
        stats_text += f"Unqualified: {100-qualified_pct:.1f}%"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.stats_dir / 'object_size_scatter.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # ========== 3. 纵横比分布 ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 整体分布
        ax = axes[0, 0]
        ax.hist(df['aspect_ratio'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', label='Square (AR=1)')
        ax.set_xlabel('Aspect Ratio (W/H)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Overall Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 按部件类型
        ax = axes[0, 1]
        for comp in df['component'].unique():
            data = df[df['component'] == comp]['aspect_ratio']
            color = np.array(COMPONENT_COLORS[comp]['base']) / 255
            ax.hist(data, bins=30, alpha=0.5, label=comp, color=color, edgecolor='black')
        ax.set_xlabel('Aspect Ratio (W/H)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Aspect Ratio by Component', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # anchor_rope 专项分析
        ax = axes[1, 0]
        rope_data = df[df['component'] == 'anchor_rope']
        if len(rope_data) > 0:
            ax.hist(rope_data['aspect_ratio'], bins=30, color='steelblue', 
                   edgecolor='black', alpha=0.7)
            ax.axvline(x=rope_data['aspect_ratio'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {rope_data["aspect_ratio"].mean():.2f}')
            ax.axvline(x=rope_data['aspect_ratio'].median(), color='green', 
                      linestyle='--', linewidth=2, label=f'Median: {rope_data["aspect_ratio"].median():.2f}')
            ax.set_xlabel('Aspect Ratio (W/H)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Anchor Rope Aspect Ratio\n(Should be >> 1 for long thin ropes)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 按状态分析
        ax = axes[1, 1]
        for state in STATES:
            state_data = df[df['state'] == state]
            if len(state_data) > 0:
                ax.hist(state_data['aspect_ratio'], bins=20, alpha=0.5, 
                       label=state, edgecolor='black')
        ax.set_xlabel('Aspect Ratio (W/H)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Aspect Ratio by State', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.stats_dir / 'aspect_ratio_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # 保存统计数据
        df.to_csv(self.stats_dir / 'annotation_statistics.csv', index=False)
        
        print(f"✅ 统计图表已保存至: {self.stats_dir}")
        
        return df
    
    def _get_color_for_bar(self, class_name: str) -> np.ndarray:
        """为柱状图获取颜色"""
        for comp, ids in COMPONENT_TYPES.items():
            if any(CLASS_NAMES[i] == class_name for i in ids):
                return np.array(COMPONENT_COLORS[comp]['base']) / 255
        return np.array([0.5, 0.5, 0.5])
    
    def visualize_hard_cases(self, max_cases: int = 30):
        """
        难例挖掘可视化
        
        Args:
            max_cases: 最大展示数量
        """
        print(f"🔍 分析难例样本...")
        
        # 提取difficult样本
        hard_cases = []
        for img_info in self.images_info:
            for anno in img_info.annotations:
                if anno.difficult:
                    hard_cases.append({
                        'image_info': img_info,
                        'annotation': anno,
                        'reason': self._analyze_hard_reason(img_info, anno)
                    })
        
        if not hard_cases:
            print("⚠️ 未发现标记为difficult的样本")
            return
        
        print(f"  发现 {len(hard_cases)} 个难例")
        
        # 按原因分类统计
        reason_counter = Counter([case['reason'] for case in hard_cases])
        print(f"  原因分布: {dict(reason_counter)}")
        
        # 生成PDF报告（多页）
        cases_to_show = hard_cases[:max_cases]
        n_pages = (len(cases_to_show) + 5) // 6  # 每页6个
        
        for page in range(n_pages):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Hard Cases Analysis (Page {page+1}/{n_pages})', 
                        fontsize=16, fontweight='bold')
            
            start_idx = page * 6
            end_idx = min(start_idx + 6, len(cases_to_show))
            
            for idx, case in enumerate(cases_to_show[start_idx:end_idx]):
                ax = axes.flat[idx]
                
                img_info = case['image_info']
                anno = case['annotation']
                reason = case['reason']
                
                # 读取并绘制
                img = cv2.imread(img_info.filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 只绘制当前难例
                h, w = img.shape[:2]
                x1 = int(anno.x_min * w)
                y1 = int(anno.y_min * h)
                x2 = int(anno.x_max * w)
                y2 = int(anno.y_max * h)
                
                color = self._get_color_for_annotation(anno)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # 添加原因标签
                label = f"[HARD] {anno.class_name}\n{reason}"
                y_text = y1 - 10 if y1 > 50 else y2 + 30
                cv2.putText(img, label, (x1, y_text), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                ax.imshow(img)
                ax.set_title(f"{img_info.filename[:25]}...\n"
                            f"Size: {anno.width*w:.0f}x{anno.height*h:.0f}px\n"
                            f"Reason: {reason}", 
                            fontsize=9, color='red')
                ax.axis('off')
            
            # 隐藏空白子图
            for idx in range(end_idx - start_idx, 6):
                axes.flat[idx].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            output_path = self.hard_cases_dir / f'hard_cases_page_{page+1:02d}.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
        
        # 生成原因汇总图
        fig, ax = plt.subplots(figsize=(10, 6))
        reasons = list(reason_counter.keys())
        counts = list(reason_counter.values())
        
        bars = ax.barh(reasons, counts, color='coral', edgecolor='black')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Hard Cases Reason Distribution', fontsize=14, fontweight='bold')
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {count}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.hard_cases_dir / 'hard_cases_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 难例分析报告已保存至: {self.hard_cases_dir}")
    
    def _analyze_hard_reason(self, img_info: ImageInfo, anno: Annotation) -> str:
        """分析难例原因"""
        h, w = img_info.height, img_info.width
        box_w = anno.width * w
        box_h = anno.height * h
        
        reasons = []
        
        # 小目标
        if box_w < 50 or box_h < 50:
            reasons.append("Small target")
        
        # 极端纵横比
        if anno.aspect_ratio > 10 or anno.aspect_ratio < 0.1:
            reasons.append("Extreme AR")
        
        # 背景复杂度
        if 'glare' in img_info.tags:
            reasons.append("Glare interference")
        if 'algae' in img_info.tags:
            reasons.append("Algae cover")
        if 'occlusion' in img_info.tags:
            reasons.append("Occlusion")
        
        # 模糊检测（基于拉普拉斯算子）
        img = cv2.imread(img_info.filepath)
        if img is not None:
            roi = img[int(anno.y_min*h):int(anno.y_max*h), 
                     int(anno.x_min*w):int(anno.x_max*w)]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 100:
                    reasons.append("Motion blur")
        
        return " + ".join(reasons) if reasons else "Unknown"
    
    def generate_html_report(self):
        """生成HTML交互式报告"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过HTML报告生成")
            return
        
        print("🌐 生成HTML交互式报告...")
        
        # 收集数据
        all_data = []
        for img_info in self.images_info:
            for anno in img_info.annotations:
                all_data.append({
                    'filename': img_info.filename,
                    'class_name': anno.class_name,
                    'component': anno.component,
                    'state': anno.state,
                    'width_px': anno.width * img_info.width,
                    'height_px': anno.height * img_info.height,
                    'area_px': anno.width * anno.height * img_info.width * img_info.height,
                    'aspect_ratio': anno.aspect_ratio,
                    'difficult': anno.difficult,
                    'filepath': img_info.filepath,
                    'tags': ', '.join(img_info.tags) if img_info.tags else 'None'
                })
        
        df = pd.DataFrame(all_data)
        
        # 创建交互式图表
        # 1. 类别分布饼图
        fig_pie = px.pie(df, names='class_name', title='Class Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        # 2. 目标尺寸散点图
        fig_scatter = px.scatter(df, x='width_px', y='height_px', 
                                color='component', symbol='state',
                                hover_data=['filename', 'class_name'],
                                title='Object Size Distribution',
                                labels={'width_px': 'Width (px)', 
                                       'height_px': 'Height (px)'})
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="red")
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="red")
        
        # 3. 部件-状态热力图
        pivot = pd.crosstab(df['component'], df['state'])
        fig_heatmap = px.imshow(pivot, text_auto=True, aspect='auto',
                               title='Component-State Heatmap',
                               color_continuous_scale='YlOrRd')
        
        # 4. 背景复杂度分布
        tag_counts = Counter()
        for img_info in self.images_info:
            for tag in img_info.tags:
                tag_counts[tag] += 1
        
        fig_tags = px.bar(x=list(tag_counts.keys()), y=list(tag_counts.values()),
                         title='Background Complexity Distribution',
                         labels={'x': 'Tag', 'y': 'Count'},
                         color=list(tag_counts.values()),
                         color_continuous_scale='Viridis')
        
        # 组合图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution', 'Object Size', 
                           'Component-State Heatmap', 'Background Complexity'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 添加饼图
        for trace in fig_pie.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 添加散点图
        for trace in fig_scatter.data:
            fig.add_trace(trace, row=1, col=2)
        
        # 添加热力图
        fig.add_trace(fig_heatmap.data[0], row=2, col=1)
        
        # 添加柱状图
        fig.add_trace(fig_tags.data[0], row=2, col=2)
        
        fig.update_layout(height=900, showlegend=False,
                         title_text="Dataset Analysis Dashboard",
                         title_font_size=20)
        
        # 保存HTML
        html_path = self.reports_dir / 'interactive_report.html'
        pyo.plot(fig, filename=str(html_path), auto_open=False)
        
        print(f"✅ HTML报告已保存: {html_path}")
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("=" * 60)
        print("🚀 开始数据集可视化分析")
        print("=" * 60)
        
        # 1. 随机抽样网格
        self.visualize_random_grid(num_samples=20, grid_size=(4, 5))
        
        # 2. 统计图表
        self.generate_statistics()
        
        # 3. 难例分析
        self.visualize_hard_cases(max_cases=30)
        
        # 4. HTML报告
        self.generate_html_report()
        
        print("=" * 60)
        print("✅ 分析完成！输出目录:")
        print(f"   网格图: {self.vis_grid_dir}")
        print(f"   统计图: {self.stats_dir}")
        print(f"   难例:   {self.hard_cases_dir}")
        print(f"   报告:   {self.reports_dir}")
        print("=" * 60)


def main():
    """主入口"""
    # 配置路径（根据实际情况修改）
    DATA_YAML = 'data/processed/data.yaml'
    IMAGES_DIR = 'data/processed/images/train'
    LABELS_DIR = 'data/processed/labels/train'
    OUTPUT_DIR = 'outputs'
    
    # 检查路径
    if not Path(IMAGES_DIR).exists():
        print(f"⚠️ 图像目录不存在: {IMAGES_DIR}")
        print("请确保已运行数据预处理脚本")
        return
    
    # 创建可视化器并运行
    visualizer = DatasetVisualizer(
        data_yaml_path=DATA_YAML,
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        min_pixel_size=50
    )
    
    visualizer.run_full_analysis()


if __name__ == '__main__':
    main()
