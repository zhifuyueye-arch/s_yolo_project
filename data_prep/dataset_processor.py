"""
水面光伏锚固系统数据集处理器
功能：数据清洗、格式转换、增强和可视化
Author: CV Engineer
Date: 2026-02-27
"""

import os
import cv2
import json
import random
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_prep.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """标注数据结构"""
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (归一化)
    segmentation: Optional[List[List[float]]] = None  # 分割点列表
    difficult: bool = False
    truncated: bool = False


class ClassMapper:
    """类别映射管理器 - 扁平化16类策略"""
    
    # 16个检测类别定义 (4 targets × 4 states)
    CLASS_NAMES = [
        'anchor_rope_normal',      # 0: 锚绳-正常
        'anchor_rope_loose',       # 1: 锚绳-松弛/磨损
        'anchor_rope_missing',     # 2: 锚绳-断裂/缺失
        'anchor_rope_damaged',     # 3: 锚绳-破损
        'anchor_connector_normal', # 4: 连接件-正常
        'anchor_connector_loose',  # 5: 连接件-松动
        'anchor_connector_missing',# 6: 连接件-脱落
        'anchor_connector_damaged',# 7: 连接件-腐蚀/破损
        'anchor_block_normal',     # 8: 锚块-正常
        'anchor_block_loose',      # 9: 锚块-移位
        'anchor_block_missing',    # 10: 锚块-沉降/缺失
        'anchor_block_damaged',    # 11: 锚块-破损
        'float_fastener_normal',   # 12: 浮筒连接件-正常
        'float_fastener_loose',    # 13: 浮筒连接件-松动
        'float_fastener_missing',  # 14: 浮筒连接件-脱落
        'float_fastener_damaged',  # 15: 浮筒连接件-破损
    ]
    
    # 类别到ID映射
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    # ID到类别映射
    ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    
    # 类别颜色映射 (用于可视化)
    CLASS_COLORS = {
        0: (0, 255, 0),      # 正常 - 绿色
        1: (0, 165, 255),    # 松弛 - 橙色
        2: (0, 0, 255),      # 缺失 - 红色
        3: (128, 0, 128),    # 破损 - 紫色
        4: (0, 255, 0),
        5: (0, 165, 255),
        6: (0, 0, 255),
        7: (128, 0, 128),
        8: (0, 255, 0),
        9: (0, 165, 255),
        10: (0, 0, 255),
        11: (128, 0, 128),
        12: (0, 255, 0),
        13: (0, 165, 255),
        14: (0, 0, 255),
        15: (128, 0, 128),
    }
    
    @classmethod
    def get_color(cls, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        return cls.CLASS_COLORS.get(class_id, (128, 128, 128))
    
    @classmethod
    def validate_class(cls, class_name: str) -> bool:
        """验证类别名称是否有效"""
        return class_name in cls.CLASS_NAMES


class DataAugmentation:
    """水面场景专用数据增强器"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: 应用增强的概率
        """
        self.p = p
    
    def apply(self, image: np.ndarray, annotations: List[Annotation]) -> Tuple[np.ndarray, List[Annotation]]:
        """应用随机增强组合"""
        if random.random() > self.p:
            return image, annotations
        
        aug_list = [
            self.random_brightness,
            self.random_contrast,
            self.color_jitter,
            self.gaussian_noise,
            self.simulate_water_ripple,
            self.simulate_glare,
            self.simulate_algae_cover,
        ]
        
        # 随机选择2-4个增强
        num_augs = random.randint(2, 4)
        selected_augs = random.sample(aug_list, num_augs)
        
        for aug in selected_augs:
            if random.random() < 0.7:  # 单个增强的概率
                image = aug(image)
        
        return image, annotations
    
    def random_brightness(self, image: np.ndarray, delta: float = 0.3) -> np.ndarray:
        """随机亮度调整"""
        factor = 1.0 + random.uniform(-delta, delta)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def random_contrast(self, image: np.ndarray, delta: float = 0.3) -> np.ndarray:
        """随机对比度调整"""
        factor = 1.0 + random.uniform(-delta, delta)
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def color_jitter(self, image: np.ndarray) -> np.ndarray:
        """颜色抖动 (模拟不同时间光照)"""
        # 调整色相
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.9, 1.1), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def gaussian_noise(self, image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def simulate_water_ripple(self, image: np.ndarray) -> np.ndarray:
        """模拟水面波纹效果"""
        h, w = image.shape[:2]
        # 创建波纹变形图
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        frequency = random.uniform(0.01, 0.03)
        amplitude = random.uniform(2, 5)
        
        for y in range(h):
            for x in range(w):
                offset_x = int(amplitude * np.sin(2 * np.pi * y * frequency))
                offset_y = int(amplitude * np.cos(2 * np.pi * x * frequency))
                map_x[y, x] = min(max(x + offset_x, 0), w - 1)
                map_y[y, x] = min(max(y + offset_y, 0), h - 1)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    def simulate_glare(self, image: np.ndarray) -> np.ndarray:
        """模拟强反光效果 (水面阳光反射)"""
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # 随机生成1-3个反光区域
        num_glares = random.randint(1, 3)
        for _ in range(num_glares):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            radius = random.randint(50, 200)
            intensity = random.uniform(0.3, 0.7)
            
            # 创建径向渐变反光
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = np.exp(-dist / radius)
            mask = np.clip(mask * intensity * 255, 0, 255)
            
            # 添加白色反光
            for c in range(3):
                result[:, :, c] = np.clip(result[:, :, c] + mask, 0, 255)
        
        return result.astype(np.uint8)
    
    def simulate_algae_cover(self, image: np.ndarray) -> np.ndarray:
        """模拟藻类覆盖效果"""
        h, w = image.shape[:2]
        # 生成噪声纹理模拟藻类
        noise = np.random.randint(0, 50, (h, w), dtype=np.uint8)
        noise_colored = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        
        # 绿色调
        algae_mask = np.zeros_like(image)
        algae_mask[:, :, 1] = random.randint(30, 80)  # 绿色通道
        
        # 随机位置叠加
        alpha = random.uniform(0.1, 0.3)
        result = cv2.addWeighted(image, 1.0, algae_mask, alpha, 0)
        
        return result
    
    def mosaic(self, images: List[np.ndarray], annotations_list: List[List[Annotation]], 
               size: int = 640) -> Tuple[np.ndarray, List[Annotation]]:
        """
        Mosaic增强 - 将4张图片拼接为一张
        适用于水面场景小目标密集检测
        """
        assert len(images) == 4, "Mosaic需要4张图片"
        
        mosaic_img = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        mosaic_annos = []
        
        # 随机中心点
        yc, xc = (int(random.uniform(size * 0.5, size * 1.5)) for _ in range(2))
        
        for i, (img, annos) in enumerate(zip(images, annotations_list)):
            h, w = img.shape[:2]
            
            # 计算放置位置
            if i == 0:  # 左上
                x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
                img_x1, img_y1, img_x2, img_y2 = w - (x2 - x1), h - (y2 - y1), w, h
            elif i == 1:  # 右上
                x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, size * 2), yc
                img_x1, img_y1, img_x2, img_y2 = 0, h - (y2 - y1), min(w, x2 - x1), h
            elif i == 2:  # 左下
                x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(yc + h, size * 2)
                img_x1, img_y1, img_x2, img_y2 = w - (x2 - x1), 0, w, min(h, y2 - y1)
            else:  # 右下
                x1, y1, x2, y2 = xc, yc, min(xc + w, size * 2), min(yc + h, size * 2)
                img_x1, img_y1, img_x2, img_y2 = 0, 0, min(w, x2 - x1), min(h, y2 - y1)
            
            # 放置图片
            mosaic_img[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]
            
            # 调整标注坐标
            for anno in annos:
                x1_norm, y1_norm, x2_norm, y2_norm = anno.bbox
                x1_abs = x1_norm * w + x1 - img_x1
                y1_abs = y1_norm * h + y1 - img_y1
                x2_abs = x2_norm * w + x1 - img_x1
                y2_abs = y2_norm * h + y1 - img_y1
                
                # 裁剪到mosaic边界
                x1_new = max(0, min(x1_abs / (size * 2), 1))
                y1_new = max(0, min(y1_abs / (size * 2), 1))
                x2_new = max(0, min(x2_abs / (size * 2), 1))
                y2_new = max(0, min(y2_abs / (size * 2), 1))
                
                if x2_new > x1_new and y2_new > y1_new:
                    mosaic_annos.append(Annotation(
                        class_name=anno.class_name,
                        bbox=(x1_new, y1_new, x2_new, y2_new),
                        difficult=anno.difficult
                    ))
        
        return mosaic_img, mosaic_annos


class AnnotationConverter:
    """标注格式转换器"""
    
    @staticmethod
    def line_to_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        将线段标注转换为最小外接矩形
        适用于anchor_rope的线段标注
        """
        if len(points) < 2:
            raise ValueError("线段至少需要2个点")
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # 最小外接矩形
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 对于细长的绳子，增加一定宽度 (约10像素归一化)
        width = x_max - x_min
        height = y_max - y_min
        
        if width < 0.01:  # 垂直线
            x_min = max(0, x_min - 0.005)
            x_max = min(1, x_max + 0.005)
        if height < 0.01:  # 水平线
            y_min = max(0, y_min - 0.005)
            y_max = min(1, y_max + 0.005)
        
        return (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def line_to_segmentation(points: List[Tuple[float, float]], width: float = 0.01) -> List[List[float]]:
        """
        将线段转换为分割掩码点序列
        用于实例分割模型
        
        Args:
            points: 线段中心点 [(x1,y1), (x2,y2), ...]
            width: 线宽 (归一化)
        """
        if len(points) < 2:
            return []
        
        # 计算垂直方向偏移生成掩码
        polygon = []
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # 计算垂直方向
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                perp_x = -dy / length * width / 2
                perp_y = dx / length * width / 2
                
                # 上边缘
                polygon.extend([
                    x1 + perp_x, y1 + perp_y,
                    x2 + perp_x, y2 + perp_y
                ])
                # 下边缘 (反向)
                polygon.extend([
                    x2 - perp_x, y2 - perp_y,
                    x1 - perp_x, y1 - perp_y
                ])
        
        return [polygon]
    
    @staticmethod
    def to_yolo_format(annotation: Annotation, img_w: int, img_h: int) -> str:
        """
        转换为YOLO格式: <class_id> <x_center> <y_center> <width> <height>
        
        Returns:
            YOLO格式字符串
        """
        class_id = ClassMapper.CLASS_TO_ID.get(annotation.class_name, -1)
        if class_id == -1:
            logger.warning(f"未知类别: {annotation.class_name}")
            return ""
        
        x1, y1, x2, y2 = annotation.bbox
        
        # 转换为YOLO格式 (中心点 + 宽高)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # 跳过过小目标
        if w * img_w < 10 or h * img_h < 10:
            logger.debug(f"目标过小，跳过: {annotation.class_name}")
            return ""
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
    
    @staticmethod
    def to_yolo_segmentation(annotation: Annotation) -> str:
        """
        转换为YOLO分割格式: <class_id> <x1> <y1> <x2> <y2> ...
        """
        if annotation.segmentation is None:
            return ""
        
        class_id = ClassMapper.CLASS_TO_ID.get(annotation.class_name, -1)
        if class_id == -1:
            return ""
        
        points_str = " ".join([f"{p:.6f}" for p in annotation.segmentation[0]])
        return f"{class_id} {points_str}"


class DatasetProcessor:
    """数据集处理器主类"""
    
    def __init__(self, 
                 raw_data_dir: str,
                 output_dir: str,
                 min_resolution: Tuple[int, int] = (1920, 1080),
                 min_target_size: int = 50,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 use_segmentation: bool = False):
        """
        Args:
            raw_data_dir: 原始数据目录
            output_dir: 输出目录
            min_resolution: 最小分辨率要求
            min_target_size: 最小目标尺寸(像素)
            train_ratio/val_ratio/test_ratio: 划分比例
            use_segmentation: 是否使用分割格式
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.min_resolution = min_resolution
        self.min_target_size = min_target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_segmentation = use_segmentation
        
        self.augmenter = DataAugmentation(p=0.5)
        self.converter = AnnotationConverter()
        
        # 统计数据
        self.stats = {
            'total_images': 0,
            'filtered_images': 0,
            'total_annotations': 0,
            'filtered_annotations': 0,
            'class_distribution': defaultdict(int),
        }
        
        # 创建输出目录
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = ['images/train', 'images/val', 'images/test',
                'labels/train', 'labels/val', 'labels/test']
        
        if self.use_segmentation:
            dirs.extend(['labels_seg/train', 'labels_seg/val', 'labels_seg/test'])
        
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
    
    def parse_custom_annotation(self, anno_path: str) -> List[Annotation]:
        """
        解析自定义格式标注文件
        支持JSON/XML格式，包含difficult字段
        
        示例JSON格式:
        {
            "objects": [
                {
                    "class": "anchor_rope",
                    "state": "loose",
                    "bbox": [x1, y1, x2, y2],
                    "difficult": false
                }
            ]
        }
        """
        annotations = []
        
        try:
            if anno_path.endswith('.json'):
                with open(anno_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for obj in data.get('objects', []):
                    class_name = f"{obj['class']}_{obj['state']}"
                    
                    if not ClassMapper.validate_class(class_name):
                        logger.warning(f"跳过无效类别: {class_name}")
                        continue
                    
                    bbox = obj.get('bbox', [0, 0, 1, 1])
                    
                    # 处理线段转矩形
                    if obj['class'] == 'anchor_rope' and 'line_points' in obj:
                        bbox = self.converter.line_to_bbox(obj['line_points'])
                    
                    anno = Annotation(
                        class_name=class_name,
                        bbox=tuple(bbox),
                        difficult=obj.get('difficult', False),
                        truncated=obj.get('truncated', False)
                    )
                    annotations.append(anno)
                    
            elif anno_path.endswith('.xml'):
                tree = ET.parse(anno_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    difficult = obj.find('difficult') is not None and obj.find('difficult').text == '1'
                    
                    bbox_elem = obj.find('bndbox')
                    if bbox_elem is not None:
                        bbox = (
                            float(bbox_elem.find('xmin').text),
                            float(bbox_elem.find('ymin').text),
                            float(bbox_elem.find('xmax').text),
                            float(bbox_elem.find('ymax').text)
                        )
                        
                        anno = Annotation(
                            class_name=class_name,
                            bbox=bbox,
                            difficult=difficult
                        )
                        annotations.append(anno)
        
        except Exception as e:
            logger.error(f"解析标注文件失败 {anno_path}: {e}")
        
        return annotations
    
    def filter_data(self, image_path: str, annotations: List[Annotation]) -> Tuple[bool, List[Annotation]]:
        """
        数据清洗：过滤低质量样本和小目标
        
        Returns:
            (是否保留, 过滤后的标注)
        """
        # 读取图像检查分辨率
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图像: {image_path}")
            return False, []
        
        h, w = img.shape[:2]
        
        # 检查分辨率
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            logger.debug(f"分辨率不足: {w}x{h} < {self.min_resolution}")
            return False, []
        
        # 过滤小目标和difficult样本
        filtered_annos = []
        for anno in annotations:
            # 跳过difficult样本 (可在配置中调整)
            if anno.difficult:
                continue
            
            x1, y1, x2, y2 = anno.bbox
            box_w = int((x2 - x1) * w)
            box_h = int((y2 - y1) * h)
            
            # 检查目标尺寸
            if box_w < self.min_target_size or box_h < self.min_target_size:
                logger.debug(f"目标过小: {box_w}x{box_h}")
                continue
            
            filtered_annos.append(anno)
        
        # 过滤无有效标注的图像
        if len(filtered_annos) == 0:
            logger.debug(f"无有效标注: {image_path}")
            return False, []
        
        return True, filtered_annos
    
    def process_dataset(self):
        """主处理流程"""
        logger.info("开始处理数据集...")
        
        # 查找所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.raw_data_dir.rglob(f'*{ext}'))
        
        logger.info(f"找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        valid_samples = []
        
        for img_path in tqdm(image_files, desc="处理图像"):
            # 查找对应标注文件
            anno_path_json = img_path.with_suffix('.json')
            anno_path_xml = img_path.with_suffix('.xml')
            
            if anno_path_json.exists():
                anno_path = str(anno_path_json)
            elif anno_path_xml.exists():
                anno_path = str(anno_path_xml)
            else:
                logger.warning(f"未找到标注文件: {img_path}")
                continue
            
            # 解析标注
            annotations = self.parse_custom_annotation(anno_path)
            self.stats['total_annotations'] += len(annotations)
            
            # 数据清洗
            keep, filtered_annos = self.filter_data(str(img_path), annotations)
            
            if keep:
                valid_samples.append({
                    'image': str(img_path),
                    'annotations': filtered_annos
                })
                self.stats['filtered_annotations'] += len(filtered_annos)
                
                # 统计类别分布
                for anno in filtered_annos:
                    self.stats['class_distribution'][anno.class_name] += 1
            else:
                self.stats['filtered_images'] += 1
        
        self.stats['total_images'] = len(image_files)
        logger.info(f"有效样本: {len(valid_samples)} / {len(image_files)}")
        
        # 划分数据集
        random.shuffle(valid_samples)
        n = len(valid_samples)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train_samples = valid_samples[:n_train]
        val_samples = valid_samples[n_train:n_train + n_val]
        test_samples = valid_samples[n_train + n_val:]
        
        logger.info(f"数据集划分 - 训练集: {len(train_samples)}, 验证集: {len(val_samples)}, 测试集: {len(test_samples)}")
        
        # 处理每个子集
        self._process_split(train_samples, 'train')
        self._process_split(val_samples, 'val')
        self._process_split(test_samples, 'test')
        
        # 生成配置文件
        self._generate_yaml()
        
        # 保存统计信息
        self._save_statistics()
        
        logger.info("数据集处理完成!")
    
    def _process_split(self, samples: List[Dict], split_name: str):
        """处理单个数据子集"""
        logger.info(f"处理 {split_name} 集...")
        
        img_output_dir = self.output_dir / 'images' / split_name
        label_output_dir = self.output_dir / 'labels' / split_name
        
        for sample in tqdm(samples, desc=f"处理{split_name}"):
            img_path = Path(sample['image'])
            annotations = sample['annotations']
            
            # 读取图像
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # 数据增强 (仅训练集)
            if split_name == 'train' and random.random() < 0.5:
                img, annotations = self.augmenter.apply(img, annotations)
            
            # 保存图像
            output_img_path = img_output_dir / img_path.name
            cv2.imwrite(str(output_img_path), img)
            
            # 转换并保存标注
            yolo_lines = []
            for anno in annotations:
                line = self.converter.to_yolo_format(anno, w, h)
                if line:
                    yolo_lines.append(line)
            
            label_file = label_output_dir / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    def _generate_yaml(self):
        """生成YOLO数据配置文件"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(ClassMapper.CLASS_NAMES),
            'names': {i: name for i, name in enumerate(ClassMapper.CLASS_NAMES)}
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        
        logger.info(f"生成数据配置文件: {yaml_path}")
    
    def _save_statistics(self):
        """保存统计信息"""
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存统计信息: {stats_path}")
    
    def visualize_dataset(self, num_samples: int = 20):
        """
        可视化数据集样本
        
        Args:
            num_samples: 可视化样本数量
        """
        logger.info(f"生成可视化样本 (n={num_samples})...")
        
        vis_output_dir = Path('results/visualized_dataset')
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从训练集中随机选择样本
        train_img_dir = self.output_dir / 'images' / 'train'
        train_label_dir = self.output_dir / 'labels' / 'train'
        
        image_files = list(train_img_dir.glob('*.jpg'))[:num_samples]
        
        for img_path in tqdm(image_files, desc="可视化"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 读取标注
            label_path = train_label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 绘制边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])
                
                # 转换为像素坐标
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)
                
                # 获取颜色
                color = ClassMapper.get_color(class_id)
                class_name = ClassMapper.ID_TO_CLASS.get(class_id, 'unknown')
                
                # 绘制
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_name, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存
            output_path = vis_output_dir / f"vis_{img_path.name}"
            cv2.imwrite(str(output_path), img)
        
        logger.info(f"可视化完成，保存至: {vis_output_dir}")


def main():
    """主入口"""
    # 配置参数
    processor = DatasetProcessor(
        raw_data_dir='data/raw',
        output_dir='data/processed',
        min_resolution=(1920, 1080),
        min_target_size=50,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        use_segmentation=False
    )
    
    # 执行处理
    processor.process_dataset()
    
    # 生成可视化
    processor.visualize_dataset(num_samples=20)


if __name__ == '__main__':
    main()
