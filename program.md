# 甄嬛 Roleplay 模型训练任务

## 任务与目标

- 使用 `dataset/` 目录下已有的数据集，基于 **Qwen3-0.6B** 训练一个扮演甄嬛的角色扮演（roleplay）模型。
- 模型必须是**非思考模型**，使用 **Apple MLX** 框架在 Apple M4 电脑上训练。
- 由 coding agent 自建甄嬛角色扮演评测标准及 **Huanhuan Roleplay Evaluation Benchmark**，用于比较训练前后的模型效果。

## 约束

- 全程使用 **uv** 管理 Python 环境及依赖。
- 使用 **ModelScope** 下载模型，并将模型保存到 `models/` 目录下。
- 使用 **SwanLab** 记录训练过程，可使用 SwanLab skill 分析训练数据与实验结果。
- 最多训练 **10 次**，每次训练时间不得超过 **20 分钟**。
- 不更换基础模型或训练框架，充分利用已有数据集和本地硬件条件。

## 评估与交付

- 详细记录每一次改进的内容及原因，明确区分提示词、训练集、训练参数或其他方面的调整。
- 训练完成后，提供**训练前模型**与**最终训练完成模型**的实际效果对比。
- 提供两个模型在自建 Huanhuan Roleplay Evaluation Benchmark 上的表现及结果分析。
- 交付最终模型、相关训练与评测记录，以及完整的改进说明。

## 参考链接

- [MLX-LM 官方 LoRA / QLoRA 微调文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [Qwen3-0.6B + MLX LoRA 示例项目](https://github.com/chalecao/finetune-qwen-0.6b-lora)
- [Qwen3-0.6B LoRA 训练配置示例](https://github.com/chalecao/finetune-qwen-0.6b-lora/blob/main/ft_qwen3_lora.yaml)
