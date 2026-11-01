# chat-huanhuan-mlx

基于 **Qwen3-0.6B** 的甄嬛角色扮演（roleplay）微调项目。
基础模型经 **ModelScope** 下载，使用 **Apple MLX** 框架在 **Apple M4 (24GB)**
上以 LoRA 方式训练，训练过程通过 **SwanLab** 记录。
模型为**非思考模型**（推理时 `enable_thinking=False`，训练目标不含 `</think>`）。

> 任务说明见 [`program.md`](program.md)（该文件只读，未作任何修改）。

## 目录结构

```
program.md                 任务书（不可修改）
dataset/huanhuan.json      原始甄嬛对白数据集（3729 条）
models/
  Qwen3-0.6B/              ModelScope 下载的原始 HF 权重
  Qwen3-0.6B-mlx/          转换后的 MLX 权重（uv run mlx_lm.convert 生成）
data/v1/                   mlx-lm chat 格式 SFT 数据（train/valid.jsonl）
benchmark/
  benchmark.json           自建 Huanhuan Roleplay Evaluation Benchmark（41 条，v1.2）
  README.md                评测标准与评分规则
configs/*.yaml             各次训练的超参配置
scripts/
  prompts.py               训练/评测共用系统提示词
  prepare_data.py          数据构建（含去重/增强选项）
  train.py                 受限训练运行器（≤10 次、每次 ≤20 分钟、SwanLab）
  train_entry.py           mlx_lm.lora 入口 + clear_cache_threshold 接线
  enforce_timeout.py       20 分钟硬超时看门狗（兼容旧 runner）
  evaluate.py              benchmark 推理 + 自动打分
  compare.py               前后结果对比
  analyze_swanlab.py       从 SwanLab 拉取各 run 指标做汇总
  report.py                生成 REPORT.md
  finalize.py              选最佳 adapter → 复制 final → 合并 → 报告
  selftest.py              评分器自检（10 项断言）
  check_deliverables.py    交付验收（逐条对照 program.md）
outputs/
  adapters/<run>/          各次训练的 LoRA adapter
  eval/<tag>.json          各次 benchmark 结果明细
logs/
  training_runs.json       全部训练运行元数据
  runN.log                 各次训练完整日志
IMPROVEMENTS.md            每次改进的内容与原因（含维度区分）
REPORT.md                  最终交付报告（前后对比 + benchmark 分析）
```

## 快速开始

```bash
# 1. 安装依赖（uv 管理）
uv sync

# 2. 下载基础模型（ModelScope）并转换为 MLX
uv run modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
uv run mlx_lm.convert --hf-path models/Qwen3-0.6B \
    --mlx-path models/Qwen3-0.6B-mlx --dtype bfloat16

# 3. 构建训练数据
uv run python scripts/prepare_data.py --out-dir data/v1

# 4. 训练前基线评测
uv run python scripts/evaluate.py --tag base

# 5. 训练（受 20 分钟 / 10 次上限保护，自动上报 SwanLab）
uv run python scripts/train.py --name run1 --config configs/base.yaml

# 6. 训练后评测与对比
uv run python scripts/evaluate.py --tag run1 --adapter outputs/adapters/run1
uv run python scripts/compare.py --a base --b run1 --markdown
```

SwanLab 项目：`huanhuan-mlx`（云端，https://swanlab.cn/@kmno4/huanhuan-mlx ）。

### 推理加载最终模型

```python
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tok = load("models/Qwen3-0.6B-huanhuan-mlx")   # 已融合 LoRA 的独立模型
sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20)  # Qwen3 官方非思考采样
mx.random.seed(42)

msgs = [{"role": "system", "content": "你是电视剧《甄嬛传》中的甄嬛。…"},
        {"role": "user", "content": "你是谁？报上名来。"}]
prompt = tok.apply_chat_template(
    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(generate(model, tok, prompt=prompt, sampler=sampler, max_tokens=128))
```

> **请勿用贪心（temp=0）解码**：0.6B 模型在贪心下容易陷入复读环
> （final 模型 2/41 题复读，官方采样器 0/41），这也是 benchmark 与
> 本项目统一采用 Qwen3 官方非思考采样的原因。详见 `REPORT.md` 第 4 节
> 「鲁棒性」与 `outputs/robustness.json`。

## 交付验收

```bash
uv run python scripts/check_deliverables.py   # 逐条对照 program.md，全绿才交付
```

## 关键设计

- **非思考**：训练目标由 Qwen3 chat 模板在 `enable_thinking=False` 下渲染，
  assistant 目标自带空 `</think>` 块；推理同样以 `enable_thinking=False` 渲染，
  从训练与推理两端杜绝思维链。
- **只训 assistant**：`mask_prompt: true`，系统提示与用户台词不计入 loss，
  避免模型学会复读 prompt。
- **提示词一致**：训练与评测共用 `scripts/prompts.py::SYSTEM_PROMPT`，
  保证评测反映模型本身而非提示词漂移。
- **可复现评测**：benchmark 全规则自动打分，无外部 judge；固定 seed=42。

## 交付物

最终模型、训练与评测记录及完整改进说明见 [`REPORT.md`](REPORT.md)。
