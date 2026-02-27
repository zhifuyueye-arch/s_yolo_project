# 水面光伏锚固系统检测项目 - Git工作流指南

## 项目概述

本指南描述如何规范地使用Git进行版本控制，管理"水面光伏区锚固系统智能检测系统"项目的开发流程。

---

## 1. 仓库初始化

### 1.1 创建新仓库

```bash
# 在项目根目录执行
cd /path/to/project_root

# 初始化Git仓库
git init

# 添加初始文件
git add .

# 创建初始提交
git commit -m "chore: initial commit - project structure setup

- Add data preparation pipeline (dataset_processor.py)
- Add model training module (train_models.py)
- Add configuration files (data.yaml, hyp.yaml)
- Add directory structure and documentation"
```

### 1.2 连接远程仓库 (可选)

```bash
# 添加远程仓库
git remote add origin https://github.com/your-org/anchor-detection.git

# 推送主分支
git push -u origin main
```

---

## 2. 分支策略 (Git Flow)

### 2.1 分支类型说明

| 分支类型 | 命名前缀 | 用途 |
|---------|---------|------|
| 主分支 | `main` | 稳定版本，可部署 |
| 开发分支 | `develop` | 日常开发集成 |
| 功能分支 | `feat/*` | 新功能开发 |
| 修复分支 | `fix/*` | Bug修复 |
| 实验分支 | `exp/*` | 实验性功能 |
| 发布分支 | `release/*` | 版本发布准备 |

### 2.2 分支创建与管理

```bash
# 创建并切换到功能分支
git checkout -b feat/data-pipeline

# 查看所有分支
git branch -a

# 切换到已有分支
git checkout feat/train-v8

# 删除本地分支 (合并后)
git branch -d feat/data-pipeline

# 强制删除分支 (未合并)
git branch -D feat/data-pipeline
```

---

## 3. 功能开发工作流

### 阶段1: 数据集构建模块 (feat/data-pipeline)

```bash
# 1. 从main创建功能分支
git checkout main
git pull origin main
git checkout -b feat/data-pipeline

# 2. 开发数据集处理功能
# - 编辑 data_prep/dataset_processor.py
# - 添加数据清洗逻辑
# - 实现增强策略

# 3. 提交更改 (遵循提交信息规范)
git add data_prep/
git commit -m "feat(data): implement data cleaning and filtering

- Add resolution filtering (min 1920x1080)
- Add target size filtering (min 50x50 pixels)
- Support difficult sample handling"

git add data_prep/
git commit -m "feat(data): implement water scene augmentation

- Add ripple simulation
- Add glare simulation
- Add algae coverage simulation
- Implement Mosaic and Mixup augmentation"

git add configs/
git commit -m "config: add data.yaml with 16-class mapping

- Define anchor_system components and states
- Configure water-specific augmentation params"

# 4. 完成功能，合并回develop
git checkout develop
git merge feat/data-pipeline --no-ff

# 5. 推送更改
git push origin develop

# 6. 删除功能分支
git branch -d feat/data-pipeline
```

### 阶段2: YOLOv8训练模块 (feat/train-v8)

```bash
# 1. 创建功能分支
git checkout develop
git checkout -b feat/train-v8

# 2. 开发训练模块
# - 实现YOLOv8n训练器
# - 添加回调函数
# - 实现模型导出

# 3. 提交更改
git add training/
git commit -m "feat(train): implement YOLOv8n baseline training

- Add ModelConfig dataclass
- Implement YOLOTrainer with callbacks
- Add automatic mixed precision (AMP) support
- Configure for edge deployment (imgsz=640)"

git add training/
git commit -m "feat(train): add model export functionality

- Support ONNX export for edge deployment
- Add TensorRT engine export option"

# 4. 合并回develop
git checkout develop
git merge feat/train-v8 --no-ff
git push origin develop
```

### 阶段3: YOLOv12训练模块 (feat/train-v12)

```bash
# 1. 创建功能分支
git checkout develop
git checkout -b feat/train-v12

# 2. 开发SOTA模型训练
# - 实现YOLOv12训练配置
# - 调整超参数 (imgsz=1280, AdamW)

# 3. 提交更改
git add training/
git commit -m "feat(train): add YOLOv12 SOTA training support

- Configure high-resolution training (1280px)
- Use AdamW optimizer for better convergence
- Add 150 epochs training schedule
- Implement mixed precision training"

git add training/
git commit -m "feat(eval): implement model comparison framework

- Add ModelComparator for metrics analysis
- Generate comparison charts and reports
- Support confusion matrix export"

# 4. 合并回develop
git checkout develop
git merge feat/train-v12 --no-ff
git push origin develop
```

### 阶段4: 评估与可视化 (feat/evaluation)

```bash
# 1. 创建功能分支
git checkout develop
git checkout -b feat/evaluation

# 2. 开发评估模块
# - 实现推理可视化
# - 添加缺陷案例展示

# 3. 提交更改
git add training/
git commit -m "feat(eval): add inference visualization tools

- Implement side-by-side model comparison
- Add defect case visualization
- Generate evaluation reports in Markdown"

# 4. 合并回develop
git checkout develop
git merge feat/evaluation --no-ff
git push origin develop
```

---

## 4. 提交信息规范 (Commit Message Convention)

### 4.1 格式规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 4.2 类型说明

| 类型 | 说明 | 示例 |
|-----|------|------|
| `feat` | 新功能 | `feat(data): add mosaic augmentation` |
| `fix` | Bug修复 | `fix(train): resolve CUDA OOM issue` |
| `docs` | 文档更新 | `docs: update training guide` |
| `style` | 代码格式 | `style: format with black` |
| `refactor` | 重构 | `refactor: simplify data loader` |
| `perf` | 性能优化 | `perf: optimize image loading` |
| `test` | 测试 | `test: add unit tests for converter` |
| `chore` | 构建/工具 | `chore: update requirements.txt` |
| `config` | 配置变更 | `config: adjust learning rate` |

### 4.3 示例

```bash
# 功能提交
git commit -m "feat(data): implement line-to-bbox conversion for anchor rope

- Convert line annotations to minimal bounding box
- Handle vertical/horizontal line edge cases
- Add padding for thin rope targets"

# 修复提交
git commit -m "fix(train): resolve label caching issue

Labels were not properly cached after augmentation.
Now using hash-based cache invalidation.

Fixes #23"

# 文档提交
git commit -m "docs: add Git workflow guide

- Branch strategy documentation
- Commit message conventions
- Feature development workflow"
```

---

## 5. 代码审查与合并

### 5.1 Pull Request 流程

```bash
# 1. 推送功能分支到远程
git push origin feat/data-pipeline

# 2. 在GitHub/GitLab创建Pull Request
# 目标分支: develop
# 标题: [FEAT] Implement data preparation pipeline

# 3. 代码审查后，合并到develop
git checkout develop
git pull origin develop
git merge feat/data-pipeline --no-ff

# 4. 推送合并结果
git push origin develop
```

### 5.2 合并冲突解决

```bash
# 1. 获取最新代码
git checkout develop
git pull origin develop

# 2. 切换到功能分支并合并
git checkout feat/data-pipeline
git merge develop

# 3. 解决冲突 (编辑冲突文件)
# 冲突标记: <<<<<<< HEAD, =======, >>>>>>> develop

# 4. 标记冲突已解决
git add <conflicted-file>

# 5. 完成合并
git commit -m "merge: resolve conflicts with develop"
```

---

## 6. 版本发布流程

### 6.1 创建发布分支

```bash
# 1. 从develop创建发布分支
git checkout develop
git checkout -b release/v1.0.0

# 2. 版本发布准备
# - 更新版本号
# - 更新CHANGELOG
# - 最终测试

git add .
git commit -m "chore(release): prepare v1.0.0"

# 3. 合并到main
git checkout main
git merge release/v1.0.0 --no-ff
git tag -a v1.0.0 -m "Release version 1.0.0"

# 4. 推送标签
git push origin main --tags

# 5. 合并回develop
git checkout develop
git merge release/v1.0.0 --no-ff
git push origin develop

# 6. 删除发布分支
git branch -d release/v1.0.0
```

---

## 7. 常用命令速查表

```bash
# ========== 基础操作 ==========
git status                    # 查看状态
git log --oneline --graph     # 查看提交历史
git diff                      # 查看修改
git stash                     # 暂存修改
git stash pop                 # 恢复暂存

# ========== 分支操作 ==========
git branch -a                 # 列出所有分支
git checkout -b <branch>      # 创建并切换分支
git merge <branch>            # 合并分支
git branch -d <branch>        # 删除分支

# ========== 撤销操作 ==========
git checkout -- <file>        # 撤销文件修改
git reset HEAD <file>         # 取消暂存
git reset --soft HEAD~1       # 撤销上次提交 (保留修改)
git reset --hard HEAD~1       # 撤销上次提交 (丢弃修改)

# ========== 远程操作 ==========
git remote -v                 # 查看远程仓库
git fetch origin              # 获取远程更新
git pull origin <branch>      # 拉取并合并
git push origin <branch>      # 推送到远程
git push -u origin <branch>   # 推送并关联

# ========== 标签操作 ==========
git tag                       # 列出标签
git tag -a v1.0 -m "msg"      # 创建标签
git push origin --tags        # 推送所有标签
```

---

## 8. 最佳实践

1. **频繁提交**: 小步快跑，每个提交只做一件事
2. **写清楚提交信息**: 帮助他人理解代码变更
3. **先拉后推**: 推送前先拉取最新代码，避免冲突
4. **使用分支**: 不要在main分支直接开发
5. **定期同步**: 定期将develop合并到功能分支
6. **代码审查**: 重要变更通过Pull Request进行审查
7. **保护分支**: 设置main分支保护，禁止直接推送

---

## 9. 相关文件

- `.gitignore`: 忽略文件配置
- `CHANGELOG.md`: 版本变更记录
- `README.md`: 项目说明文档
