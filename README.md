# MENA 文化视觉问答（VQA）大模型微调与强化学习项目

本项目基于 Qwen-VL 4B 多模态大模型，针对中东与北非（MENA）文化场景的视觉问答任务，进行了 SFT（监督微调）与 GRPO（强化学习）训练。最终使用官方评分脚本（`score.py`）通过 BERTScore-F1 指标进行性能评估。

## 🛠️ 环境要求

多模态与强化学习（GRPO）对底层 CUDA 算子要求极高，推荐使用 AutoDL RTX 4090 实例，已验证的稳定环境如下：

- **Python**: 3.11
- **PyTorch**: 2.9.1+cu128 (必须严格匹配 CUDA 版本)
- **关键依赖**:
  - `unsloth`
  - `transformers`
  - `trl`
  - `bert-score`
  - `sacrebleu` (评测用)

## 📂 目录结构

- `SFT/`：包含监督微调脚本及多线程图像预处理逻辑。
- `强化微调/`：包含 GRPO 强化学习脚本、自定义 BERTScore 奖励函数。
- `score.py`：官方评测脚本（依赖 `backbone.py` 和 `sacrebleu`）。

## 📊 数据说明

- **训练集**：`train_en.parquet` (共 `[待填入]` 条，SFT 阶段使用全量，GRPO 阶段采样 4000 条)。
- **验证集**：`dev_en.parquet` (共 100 条)。
- **图片路径**：数据集中 `image` 列仅存储相对路径，实际图片位于 `/hy-tmp/mmculture/dataset/images_mena_train`。

## 🚀 训练流程与核心参数

### 1. SFT (监督微调) 阶段
- **核心配置**：
  - `batch_size=4`, `gradient_accumulation_steps=8`
  - `learning_rate=2e-4`, `max_seq_length=1200`
  - 微调模块：`finetune_vision_layers=True`, `finetune_language_layers=True`
- **技巧**：图像通过 `ThreadPoolExecutor` 多线程提前加载并构建 `Dataset`，避免使用 `w.map` 导致的 Arrow 序列化死锁。
- **输出**：LoRA 适配器权重。

### 2. GRPO (强化学习) 阶段
- **核心配置**：
  ```python
  learning_rate = 5e-6         # 强化学习需极小学习率，防模型灾难性遗忘
  per_device_train_batch_size = 2
  gradient_accumulation_steps = 8
  num_generations = 4          # 同问题采样数，平衡显存与组内对比
  max_prompt_length = 1100     # 确保 900px 图像 token 不被截断
  max_completion_length = 100
  max_steps = 500              # 0.5 Epoch，防止小数据集过拟合
  
奖励函数 (Reward Function)：
使用 jhu-clsp/mmBERT-base 模型计算 BERTScore F1（提取第 17 层特征）。
关键点：奖励模型强制运行在 device="cpu"，防止与策略模型抢夺显存。
前 Baseline 分数：

SFT 模型：BERTScore-F1 = 0.8722

GRPO 模型：BERTScore-F1 = [待填入]训练时长
