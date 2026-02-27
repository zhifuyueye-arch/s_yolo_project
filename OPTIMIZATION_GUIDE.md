# 水面光伏锚固系统检测 - 特别优化建议

## 针对水面反光和小目标检测的优化策略

---

## 1. 水面反光问题优化

### 1.1 数据增强策略

```python
# 已在 dataset_processor.py 中实现
class DataAugmentation:
    def simulate_glare(self, image: np.ndarray) -> np.ndarray:
        """
        模拟强反光效果
        在训练集中添加1-3个随机位置/强度的反光区域
        提升模型对高亮区域的鲁棒性
        """
```

**建议配置参数：**
- `glare_simulation: 0.2` - 20%概率应用反光模拟
- `glare_intensity: 0.3-0.7` - 反光强度范围
- `glare_radius: 50-200px` - 反光区域大小

### 1.2 图像预处理

```python
# 建议在实际推理前添加预处理
def reduce_glare(image: np.ndarray) -> np.ndarray:
    """
    反光抑制预处理
    """
    # 方法1: CLAHE局部直方图均衡化
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 方法2: 高亮区域掩码抑制
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # 使用inpainting修复过亮区域
    result = cv2.inpaint(image, bright_mask, 3, cv2.INPAINT_TELEA)
    return result
```

### 1.3 偏振镜效果模拟

```python
def simulate_polarizer(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    模拟偏振镜效果 - 抑制水面反光
    通过降低高亮区域饱和度实现
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 识别高亮区域
    brightness_mask = hsv[:, :, 2] > 200
    
    # 降低高亮区域的饱和度 (模拟偏振效果)
    hsv[:, :, 1] = np.where(brightness_mask, 
                            hsv[:, :, 1] * (1 - strength), 
                            hsv[:, :, 1])
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### 1.4 多曝光融合 (HDR)

对于极端反光场景，建议采集多曝光图像：
```python
def fuse_exposures(images: List[np.ndarray]) -> np.ndarray:
    """
    多曝光图像融合
    适用于无人机可调节曝光参数的采集场景
    """
    merge_mertens = cv2.createMergeMertens()
    fused = merge_mertens.process(images)
    return np.clip(fused * 255, 0, 255).astype(np.uint8)
```

---

## 2. 小目标检测优化

### 2.1 高分辨率输入策略

| 模型 | 输入尺寸 | 适用场景 | 预期效果 |
|-----|---------|---------|---------|
| YOLOv8n | 640×640 | 边缘设备 | 速度优先 |
| YOLOv8n | 1280×1280 | 精度优先 | 小目标+5-8% |
| YOLOv12 | 1280×1280 | 高精度 | 小目标最佳 |
| YOLOv12 | 1920×1920 | 极致精度 | 显存受限 |

**配置建议 (configs/hyp.yaml):**
```yaml
# 高分辨率训练配置
high_res:
  imgsz: 1280
  batch: 4  # 根据显存调整 (RTX 3090可设8)
  rect: true  # 矩形训练减少padding
```

### 2.2 多尺度训练 (MST)

```python
# 在训练配置中启用多尺度
# Ultralytics YOLO自动支持，通过 scale 参数控制

# 建议设置
augmentation:
  scale: 0.9  # 缩放范围 [0.1, 1.9]，增强小目标多样性
```

### 2.3 特征金字塔优化

对于YOLOv12等先进模型，关注以下改进：

```python
# YOLOv12可能包含的优化 (基于2026年架构)
- 使用 BiFPN (Bidirectional FPN) 替代传统FPN
- 添加 P2 检测头 (用于4×4像素级小目标)
- 引入注意力机制 (CA, CBAM) 增强小目标特征
```

### 2.4 标签分配策略

```python
# 针对锚绳等细长目标，建议使用TaskAlignedAssigner
# 已在Ultralytics YOLOv8+中默认启用

# 自定义配置
training:
  tal_topk: 13           # TaskAlignedAssigner top-k
  anchor_t: 4.0          # 锚框倍数阈值
  anchor_multiple: 4.0   # 锚框长宽比阈值
```

---

## 3. 锚绳特殊处理

### 3.1 线段转检测框优化

```python
# 在 dataset_processor.py 中已实现
class AnnotationConverter:
    @staticmethod
    def line_to_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        锚绳线段转边界框的关键优化：
        1. 保留细长特征 (避免过度扩展)
        2. 添加上下文padding (10-20像素)
        3. 处理倾斜角度
        """
```

### 3.2 使用分割模型 (可选升级)

对于极细锚绳，建议升级到实例分割：

```python
# 使用 YOLOv8-seg 或 YOLOv12-seg
from ultralytics import YOLO

# 分割模型可提供更精确的锚绳掩码
model = YOLO('yolov8n-seg.pt')

# 修改数据配置
use_segmentation: true  # 在 DatasetProcessor 中启用
```

### 3.3 锚绳连接关系推理

```python
def infer_rope_connections(detections: List[Dict]) -> List[Dict]:
    """
    基于几何关系推理锚绳连接关系
    用于验证检测结果的合理性
    """
    connections = []
    for i, det1 in enumerate(detections):
        for det2 in detections[i+1:]:
            if is_valid_connection(det1, det2):
                connections.append({
                    'from': det1['id'],
                    'to': det2['id'],
                    'distance': calculate_distance(det1, det2),
                    'angle': calculate_angle(det1, det2)
                })
    return connections
```

---

## 4. 模型优化技术

### 4.1 知识蒸馏 (可选)

```python
# 使用YOLOv12作为教师模型，YOLOv8n作为学生模型
class KnowledgeDistillation:
    """
    知识蒸馏实现
    将高精度模型(YOLOv12)的知识迁移到轻量模型(YOLOv8n)
    """
    def __init__(self, teacher_model, student_model, alpha=0.5, temperature=3.0):
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha          # 硬标签权重
        self.temperature = temperature  # 温度系数
    
    def compute_loss(self, images, targets):
        # 软目标 (教师模型输出)
        with torch.no_grad():
            teacher_preds = self.teacher(images) / self.temperature
        
        # 学生模型输出
        student_preds = self.student(images) / self.temperature
        
        # 蒸馏损失 + 检测损失
        distill_loss = F.kl_div(
            F.log_softmax(student_preds, dim=-1),
            F.softmax(teacher_preds, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        detection_loss = self.student.compute_loss(images, targets)
        
        return self.alpha * detection_loss + (1 - self.alpha) * distill_loss
```

### 4.2 模型剪枝 (边缘部署)

```python
# 使用 Torch-Pruning 等工具
import torch_pruning as tp

def prune_model(model, example_inputs, pruning_ratio=0.3):
    """
    结构化剪枝，减少模型参数量
    适用于边缘设备部署
    """
    # 重要性评估
    imp = tp.importance.MagnitudeImportance(p=2)
    
    # 迭代剪枝
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
    )
    
    pruner.step()
    return model
```

### 4.3 量化加速

```python
# TensorRT INT8 量化 (需要校准数据集)
from ultralytics import YOLO

model = YOLO('best.pt')

# 导出TensorRT engine (INT8)
model.export(
    format='engine',
    imgsz=640,
    half=False,
    int8=True,
    data='data/processed/data.yaml',  # 校准数据集
    workspace=4  # GB
)
```

---

## 5. 部署优化

### 5.1 TensorRT优化

```python
# TensorRT导出配置 (configs/hyp.yaml)
export:
  format: engine
  half: true        # FP16
  int8: false       # 需要校准时启用
  workspace: 4      # 工作空间 (GB)
  dynamic: true     # 动态输入尺寸
  simplify: true    # 简化ONNX
```

### 5.2 OpenVINO优化 (Intel设备)

```python
# OpenVINO导出
model.export(format='openvino', imgsz=640, half=True)

# 使用OpenVINO Runtime
from openvino.runtime import Core

core = Core()
model = core.read_model('best_openvino_model/best.xml')
compiled_model = core.compile_model(model, 'AUTO')
```

### 5.3 ONNX Runtime优化

```python
import onnxruntime as ort

# GPU推理
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    'best.onnx',
    sess_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

---

## 6. 实际部署建议

### 6.1 边缘设备选型

| 设备 | 算力 | 推荐模型 | 预期FPS |
|-----|-----|---------|--------|
| Jetson Nano | 0.5 TFLOPS | YOLOv8n @ 416 | 15-20 |
| Jetson Orin | 5 TFLOPS | YOLOv8n @ 640 | 60+ |
| Jetson AGX | 10 TFLOPS | YOLOv8s @ 640 | 50+ |
| RK3588 | 6 TFLOPS | YOLOv8n @ 640 | 30+ |
| 服务器GPU | 100+ TFLOPS | YOLOv12 @ 1280 | 30+ |

### 6.2 多模型级联策略

```python
class CascadeDetector:
    """
    级联检测器
    1. 使用轻量模型快速筛选ROI
    2. 使用高精度模型精细检测
    """
    def __init__(self):
        self.fast_model = YOLO('yolov8n.pt')  # 快速检测
        self.accurate_model = YOLO('yolov12.pt')  # 精确检测
    
    def detect(self, image):
        # 第一阶段: 快速检测获取ROI
        fast_results = self.fast_model(image, conf=0.3)
        rois = self.extract_rois(fast_results)
        
        # 第二阶段: 精细检测
        all_detections = []
        for roi in rois:
            crop = image[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
            accurate_results = self.accurate_model(crop, conf=0.25)
            all_detections.extend(self.adjust_coordinates(accurate_results, roi))
        
        return all_detections
```

### 6.3 持续学习策略

```python
# 在线学习/增量学习
class IncrementalLearner:
    """
    增量学习框架
    适应新场景无需重新训练整个模型
    """
    def __init__(self, base_model_path):
        self.model = YOLO(base_model_path)
        self.replay_buffer = []  # 回放缓冲区
    
    def add_samples(self, new_images, new_labels):
        """添加新样本到缓冲区"""
        self.replay_buffer.extend(zip(new_images, new_labels))
        # 限制缓冲区大小
        if len(self.replay_buffer) > 1000:
            self.replay_buffer = self.replay_buffer[-1000:]
    
    def update_model(self, epochs=5):
        """增量更新模型"""
        # 混合新旧数据进行微调
        mixed_data = self.prepare_mixed_dataset()
        self.model.train(data=mixed_data, epochs=epochs, lr0=0.0001)
```

---

## 7. 性能监控与调优

### 7.1 关键指标监控

```python
# 建议监控的指标
def evaluate_model(model, test_data):
    metrics = {
        # 基础指标
        'mAP50': 0,
        'mAP5095': 0,
        
        # 小目标专项指标
        'mAP_small': 0,      # 小目标 (< 32x32)
        'mAP_medium': 0,     # 中等目标 (32x32 - 96x96)
        'mAP_large': 0,      # 大目标 (> 96x96)
        
        # 水面场景专项指标
        'mAP_glare': 0,      # 反光场景下的性能
        'mAP_shadow': 0,     # 阴影场景下的性能
        
        # 推理性能
        'fps': 0,
        'latency_ms': 0,
    }
    return metrics
```

### 7.2 自动超参调优

```python
# 使用Optuna进行超参搜索
import optuna

def objective(trial):
    # 定义搜索空间
    imgsz = trial.suggest_categorical('imgsz', [640, 960, 1280])
    lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True)
    box = trial.suggest_float('box', 5.0, 10.0)
    
    # 训练模型
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='data/processed/data.yaml',
        imgsz=imgsz,
        epochs=50,
        lr0=lr0,
        box=box,
    )
    
    # 返回验证mAP
    return results.results_dict['metrics/mAP50-95']

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best mAP: {study.best_value}")
```

---

## 8. 总结

### 推荐实施路线图

```
Phase 1 (当前): 基础实现
├── YOLOv8n baseline (640px)
├── 基础数据增强
└── 标准评估流程

Phase 2 (1-2周后): 精度优化
├── YOLOv12 SOTA模型 (1280px)
├── 反光/小目标专项增强
├── 高分辨率训练
└── 模型对比分析

Phase 3 (3-4周后): 部署优化
├── TensorRT导出
├── INT8量化
├── 边缘设备适配
└── 性能压测

Phase 4 (持续): 迭代优化
├── 在线数据收集
├── 增量学习
├── 错误案例分析
└── 模型更新
```

---

*文档更新日期: 2026-02-27*
