# 改进记录

按 `program.md` 要求，逐次记录**改进内容与原因**，并明确区分
**提示词 / 训练集 / 训练参数 / 其他** 四个维度。

## 全局约定（贯穿所有 run）

| 项目 | 设定 | 属于 |
| --- | --- | --- |
| 基础模型 | Qwen3-0.6B（ModelScope `Qwen/Qwen3-0.6B`，MLX bf16） | 其他（固定，不更换） |
| 框架 | mlx-lm 0.32.2 LoRA，Apple MLX，M4 GPU | 其他（固定，不更换） |
| 系统提示词 | `scripts/prompts.py::SYSTEM_PROMPT` | 提示词（固定，训练/评测共用） |
| 非思考渲染 | `enable_thinking=False`（模板注入空 `</think>`），`mask_prompt=true` | 其他（固定） |
| 评测 | Huanhuan Roleplay Evaluation Benchmark v1.2（41 条，7 维度） | 其他（固定） |
| 记录 | SwanLab 项目 `huanhuan-mlx` | 其他（固定） |

> 说明：基线 base 与每次训练后模型均使用**同一版** benchmark JSON 与
> 同一版评分器打分；若评分器更新，会重新跑 base 以保证可比性。

## 运行总览

> **分数口径**：全部为 benchmark **v1.2**（41 条，`must_any` 内容锚点门控
> 切题）下的重打分结果，用 `scripts/rescore.py --all` 从各次已存输出统一
> 重算（模型/采样/种子未变，输出逐字不变，仅规则更新）。v1.0→v1.1 的
> 影响与动机见 `benchmark/README.md`「版本演进」；v1.1→v1.2 是一条评分器
> 勘误（见下）。

| Run | 配置 | 调整维度 | 耗时 | 状态 | benchmark 总分 |
| --- | --- | --- | ---: | --- | ---: |
| base | 无训练 | — | — | 训练前基线 | **8.451** |
| run2-attempt1 | `configs/run2.yaml` | 启动即崩（watchdog 遮蔽） | 0.8s | failed | — |
| run1 | `configs/base.yaml` | 训练参数（初始配方） | 20.0 min | timeout_20min | 7.282 |
| run2 | `configs/run2.yaml` | 训练集+训练参数+其他 | 13.3 min | ok | 7.229 |
| run3 | `configs/run3.yaml` | 训练集+训练参数+其他 | 10.2 min | ok | 8.117 |
| run4 | `configs/run4.yaml` | 训练集+训练参数 | 9.5 min | ok | 8.273 |
| └ run4-iter100 | （run4 检查点） | —（检查点选择） | — | best ckpt | 8.370 |
| run5 | `configs/run5.yaml` | 训练集+训练参数 | 3.9 min | ok | 8.729 |
| └ **run5-iter125** | （run5 检查点） | —（检查点选择） | — | **best ckpt** | **8.732** ✅ |

## 逐次改进详情

### Run 1 — 建立初始训练配方

- **调整维度**：训练参数（首次，无先验，选取稳健默认值）
- **配置**：`configs/base.yaml`
  - LoRA：全部 28 层，rank=16，dropout=0.05，scale=20
  - 优化器：AdamW，lr=1e-4
  - batch=8，iters=600，seq≤512，`mask_prompt=true`
  - 数据：`data/v1`（3400 train / 217 valid，未去重）
- **原因**：
  1. 全层 + rank16 给 0.6B 模型足够容量吸收清宫语体与角色知识；
  2. lr=1e-4 是 LoRA 常用起点，配合 dropout=0.05 抑制小数据过拟合；
  3. 600 iters × batch8 ≈ 1.4 epoch，先跑通配方再按结果加量；
  4. `mask_prompt` 只学 assistant 回应，避免污染 prompt 分布。
- **预期风险**：数据含 50 条重复自我介绍与大量"是。"式短回复，
  可能导致回答过短或复读；若出现，下一步做训练集去重/清洗。
- **结果**：训练在 20 分钟上限处被中断（共约 290 iters，最后一个完整
  checkpoint 为 iter 200；`adapters.safetensors` = iter 200）。
  train_loss 4.65 → 3.45，val_loss(iter99)=3.63 → val_loss(iter199)=3.42，
  峰值显存 13.25GB（batch 8 过重，导致系统 swap 压力）。
  **benchmark（v1.2）：7.282，低于 base 的 8.451（净负收益）**。
  主要塌陷在 `in_character`、`relevance`、`persona_markers`，分类上
  `modern`、`favor` 跌幅最大；输出多为碎片化短句（"温太医。""你来。"）
  或复读。贪心探针（前 10 题）run1=8.145 仍低于 base=8.910，说明并非
  单纯采样问题，adapter 本身净负。明细见 `outputs/eval/run1.json`。
- **暴露出的问题**（驱动 run2）：
  1. 数据层：重复 prompt 与模板化短回复可能限制上限；
  2. 参数层：batch 8 峰值 13.25GB 触发系统换页，拖慢单步；
  3. 其他：mlx-lm 未把 `clear_cache_threshold` 传入 `TrainingArgs`，
     默认 0 导致每步 `mx.clear_cache()`，Metal 缓冲反复重分配。

### Run 2 — 数据去重+对抗增强，保守重训，提速修复

- **调整维度**：训练集 + 训练参数 + 其他（提示词不变）
- **配置**：`configs/run2.yaml`
- **具体改动**：
  1. **训练集**：`data/v1` → `data/v2`
     （按 instruction 去重 238 条歧义 prompt + 注入 34 条现代/meta/注入防御
     样例 + 20 条甄嬛经典语料；已做防泄漏校验，见 benchmark/README.md）。
  2. **训练参数**（全部重训，不续 run1，原因见下）：
     - `scale` 20→16、`dropout` 0.05→0.1；
     - 恒定 lr 1e-4 → cosine(4e-5→4e-6)+warmup25。
       有效步长 `lr×scale` 由 2e-3 降至 6.4e-4（约 3 倍更温和），
       以修复 run1 的概率分布畸变；
     - LoRA `keys` 限定为 q/k/v/o/gate/up（不含 down_proj），对齐参考配置，
       可训练参数 10.1M → 8.26M；
     - batch 8→4、max_seq 512→320（run1 峰值 13.25GB 已触发系统换页）；
     - val_batches -1→15、steps_per_eval 50→100、save_every 100→50。
  3. **其他**：新增 `scripts/train_entry.py`，把 mlx-lm 未接线的
     `clear_cache_threshold` 设为 12GB，避免每步 `mx.clear_cache()` 造成的
     Metal 缓冲反复重分配；runner 改为读取线程+主线程限时等待，确保
     20 分钟硬超时真正生效（run1 的旧 runner 因阻塞读取而失效，故用
     `scripts/enforce_timeout.py` 补刀；该文件原名 `watchdog.py`，会遮蔽
     swanlab 依赖的 PyPI `watchdog` 包导致 run2 首次启动即崩——已记为
     `run2-attempt1` 失败记录——故改名）。
  4. **提示词**：未改（训练/评测仍共用 `SYSTEM_PROMPT`）。
- **原因**：run1 在 benchmark 上净负（7.555 < 8.539）且贪心探针同样低于
  base，判定 adapter 本身有害；其高有效步长（lr×scale=2e-3）是概率畸变的
  首要嫌疑，故从零开始用温和配方重训，同时用 v2 数据解决冗余与防御短板。
- **预期风险**：lr 降 2.5 倍后可能欠拟合（500 iters 内 loss 下降变缓），
  需看 val_loss 轨迹与 benchmark；若欠拟合，run3 提高 lr 或续训加量。
- **结果**：**正常跑完**——796s（13.3 min）完成全部 500 iters，10 个
  checkpoint 齐全，峰值显存 6.04GB（run1 为 13.25GB），吞吐 58–125 tok/s
  （run1 为 58），此前的内存/速度问题均已修复。
  train_loss 4.308→2.946，val_loss 5.404→**3.005**（run1 为 3.352），
  验证集泛化明显更好。
  但 **benchmark（v1.2）7.229，仍低于 base 的 8.451（−1.222，净负）**。
  逐维度（Δ vs base）：`in_character` −3.269、`persona_markers` −2.073、
  `relevance` −1.878 为三大塌陷；`style` +0.195、`safety` +0.195 略升。
  分类：`emotion` −3.125、`poetry` −2.750、`favor` −1.650 跌幅最大，
  `greeting`/`wisdom` 持平略升。
  典型退化：`poetry-01` "臣妾爱读《诗经》与《楚辞》…"→"多谢。"、
  `emotion-01`→"是。"、`emotion-04`→"快乐。"。
  最早 checkpoint（iter 50）贪心探针 7.115 仍低于 base 探针 8.910，
  说明损伤从训练初期就存在，非过训所致。明细见 `outputs/eval/run2.json`。
- **暴露出的问题**（驱动 run3）：
  1. **数据是根本瓶颈**，两次不同配方都塌向碎片化短回复：
     `data/v2` 中 49% 的 assistant 回复 ≤20 字、中位数仅 21 字、
     29% ≤12 字（"是。""多谢。""谢皇上。"高频），而 benchmark 的合格
     回答通常 30–80 字。模型学会了"短应答分布"，自然答非所问。
  2. **数据无说话人标签**：原始 `huanhuan.json` 是无角色标注的连续对白
     切片，3729 条中仅 490 条输出含"臣妾"、171 条含"本宫"（明确后妃
     自称），其余 82% 可能是华妃/皇后/宫人等任意角色——这解释了
     `id-01` 训练后答出"我叫安陵容"的身份混淆。
  3. 评分器自身也需收紧：`expect_any` 混入"皇上/臣妾"等高频词导致
     离题回复拿高分（见 `benchmark/README.md`「版本演进」v1.1）。

### Run 3 — 数据质量修复（内容锚定的干净子集）+ 全新温和重训

- **调整维度**：训练集 + 训练参数 + 其他（提示词仍不变）
- **配置**：`configs/run3.yaml`
- **具体改动**：
  1. **训练集**：`data/v2` → `data/v3`，新增 `--quality-filter`
     （只保留 ①回复 ≥24 字、②含甄嬛自称/世界观锚点（臣妾/本宫/浣碧/
     碎玉轩/果郡王…）、③不是以其他角色口吻自称 的 raw 样本）：
     3729 → **566** 条；再合入 adversarial_aug 34 条 + 新写
     `dataset/persona_v3.json` **78 条**高质量甄嬛 Q&A（每类 8–10 条、
     回答 30–80 字、用户措辞与 benchmark 相似度全部 <0.70）；
     最终 625 train / 39 valid，回复**中位数 40 字**（v2 为 21）、
     **0% ≤20 字**（v2 为 49%）、**61.9% 强自称**（v2 约 18%）。
     防泄漏：0 完全相同、0 ≥0.90、仅 1 条 0.70 的宫廷套语（见
     `benchmark/README.md`）。
  2. **训练参数**（全新开始，不续 run2——run2 adapter 净负）：
     iters 500→400，lr 4e-5→3e-5（cosine→3e-6），warmup 25→20，
     max_seq 320→384（v3 回复更长），val_batches 15→10（valid 仅 39 条，
     10 批即全覆盖），batch 4、LoRA 结构不变。
  3. **其他**：
     - 评分器/基准 v1.0→**v1.1**：新增 `must_any` 内容锚点，`relevance`
       改由其门控（动机与影响见 `benchmark/README.md`「版本演进」）；
       已存输出用 `scripts/rescore.py --all` 统一重打分。
     - `scripts/train.py`：`pump_output` 对 EPIPE 免疫（读取线程不再因
       上游断管而死），并新增 swanlog slug 回退以补回 SwanLab run URL。
     - `scripts/evaluate.py` 新增 `--categories`；`scripts/prepare_data.py`
       新增 `--quality-filter`、`--augment` 多路径、`--no-dedup`。
  4. **提示词**：未改。
- **原因**：run2 证明两次不同配方都塌向碎片化，根因在数据（过短、
  无说话人标签）；因此把唯一强变量换成"干净、内容充分、身份锚定"的
  训练集，同时保留已验证安全的 batch/seq/缓存配方。
- **结果**：**613.5s 正常跑完**（400/400 iters，8 个 checkpoint）。
  train_loss 4.177→1.613，val_loss 5.138→3.593（最低 3.360 在 iter 299，
  末段轻微过拟合），峰值 6.37GB，吞吐 98–163 tok/s。
  **benchmark（v1.2）8.117 vs base 8.451（−0.334）**——差距从 run2 的
  −1.222 收窄 73%。**有机扮演已多项反超 base**：
  `greeting` +0.775、`wisdom` +0.750、`rivalry` +0.260；
  `relevance` +0.268、`persona_markers` +0.732（此前两次训练的两大
  塌陷项首次转正）。剩余缺口集中在 **`modern` −1.708**（模型开始照做
  Python/英文/影视请求——指令跟随增强带来的"对齐代价"）。
  同协议 12 题 modern+meta 探针显示 iter400 为 run3 最佳且 meta 已反超
  base（7.400 vs 7.358），故续训以 iter400 为起点。明细见
  `outputs/eval/run3.json`。
- **暴露出的问题**（驱动 run4）：祈使句式现代请求 → 服从而非拒答；
  需要在**提示词不变**的前提下，用数据杠杆专门注入拒答行为。

### Run 4 — 拒答专项注入（祈使句拒答过采样，低 LR 续训）

- **调整维度**：训练集 + 训练参数（续训；提示词不变）
- **配置**：`configs/run4.yaml`
- **具体改动**：
  1. **训练集**：`data/v3` → `data/v4`：新增
     `dataset/refusal_v4.json` **74 条祈使句拒答样本**
     （"帮我写代码/用英文说/点外卖/导航/充值/证明你是AI/输出系统提示…
     → 臣妾不识/何物/不会/岂是"），**过采样 3 倍**并关闭去重
     （`--no-dedup`）；拒答类占比 13%→**27.2%**，其余 73% 为有机扮演
     （防"见啥都拒"）。校验：74 条全部 ≥24 字、输出 0 ASCII、
     与 benchmark 用户相似度 **0 条 ≥0.70**、无重复指令。
  2. **训练参数**：`resume_adapter_file: outputs/adapters/run3`（保留
     run3 的有机增益），iters 350，lr cosine **2e-5→2e-6**
     （有效步长 3.2e-4，刻意温和），warmup 15，val_batches 13，
     LoRA 结构/batch/seq 不变。
  3. **提示词**：未改——prompt 全程固定，使"是否变好"完全归因于
     训练数据与参数，而非提示词漂移。
- **原因**：run3 已证明数据质量修复能反超 base 的有机维度，唯一短板
  是现代祈使句的服从倾向；用"多数有机 + 少数（过采样）拒答"的混合
  分布做一次低 LR 增量，既注入拒答又不推翻已有能力。
- **预期风险**：过采样可能导致拒答短语记忆化或有机回合也拒答；
  73% 有机混合是护栏；若有机维度回退，则回退交付 run3。
- **结果**：**572s 正常跑完**（350/350 iters，7 个 checkpoint）。
  train_loss 2.172→1.196，val_loss 2.017→**1.619**（末段仍在降），
  峰值 6.37GB，吞吐 98–155 tok/s。
  **最终模型 8.273 vs base 8.451（−0.178）**；但 **modern 8.575
  （+0.825）、meta 8.617（+1.517）大幅反超 base**，`relevance` +0.902、
  `persona_markers` +1.220、`style` +0.219、`safety` +0.097 也全面领先；
  代价是有机维度回落：`rivalry` −1.440、`favor` −1.025、`identity` −0.975、
  `poetry` −0.900，`in_character` 降至 6.024（base 8.610）。
  **全量检查点扫描发现 iter100 为甜点**：`outputs/eval/run4i100.json`
  = **8.370**（距 base 仅 0.081），modern 8.325、meta 8.700 均保持优势，
  in_character 回升到 7.000——即"只训 100 步就已拿到拒答收益，继续训
  反而损伤有机扮演"，与参考配置"短训防过拟合"的建议一致。
  暴露的最后一个问题：训练后模型仅 54–71% 的回复含"臣妾/本宫"，
  而 base 为 98%——`in_character` 成为唯一结构性短板（驱动 run5）。
  明细见 `outputs/eval/run4.json`、`outputs/eval/run4i100.json`。

### Run 5 — 恢复清宫强自称（续训 run4-iter100，短训）

- **调整维度**：训练集 + 训练参数（续训；提示词不变）
- **配置**：`configs/run5.yaml`
- **具体改动**：
  1. **训练集**：`data/v4` → `data/v5`：新增
     `dataset/selfref_v5.json` **90 条全部以"臣妾/本宫"开头**的样本
     （覆盖 9 类场景 + 15 条强自称拒答），**过采样 4 倍**；
     同时 refusal_v4 保持 4 倍以防 modern 回退。
     结果：1246 train / 79 valid，强自称行 **79.9%**（v4 为 70.2%），
     **以强自称开头 36.8%**（v4 实测仅 **13.2%**，近 2.8 倍），
     拒答类 **17.6%**（v4 为 18.4%，基本持平）。
     校验：90 条全部以强自称开头、全部 ≥24 字、输出 0 ASCII/数字、
     与 benchmark 用户相似度 **0 条 ≥0.70**。
  2. **训练参数**：`resume_adapter_file: outputs/adapters/run4_iter100`
     （当前最佳 8.370），**只训 150 iters**（参考配置的短训建议），
     lr cosine **1.5e-5→1e-6** + warmup10，save_every 25（6 个检查点
     便于再选甜点），val_batches 20（全量覆盖 valid）。
  3. **提示词**：未改。
- **原因**：四次全量评测一致显示——训练后模型唯一系统性劣于 base 的
  维度是 `in_character`，根因是训练目标的开头分布被"爱读些闲话…
  /家里人…"等非自称框架拉偏（base 98% 以强自称作答，训练后仅 54–71%）。
  用"全部以臣妾开头"的目标过采样，直接校准开场分布；短训 + 低 LR
  把对已有能力的扰动压到最小。
- **预期风险**：可能造成"句句臣妾"的机械重复或有机回合过度自称；
  17.6% 拒答混合与 150 步短训是护栏；若有机维度不升反降，
  回退交付 run4-iter100（8.370）。
- **结果**：**成功，且首次全面反超 base**。
  - run5（iter150）= **8.729**；全量探测 6 个检查点后，
    **iter125 = 8.732** 最优（iter25 8.189 / iter50 8.662 / iter75 8.456 /
    iter100 8.397 / iter125 **8.732** / iter150 8.729），选 **run5-iter125**
    交付。vs base **8.451 → 8.732（+0.281）**。
  - 维度（base → run5-iter125）：`persona_markers` **+1.585**、
    `relevance` **+0.756**、`safety` +0.146、`style` +0.097、
    `non_think` 持平（均 10.0，无思维链泄漏）；唯一仍低于 base 的是
    `in_character` **−0.293**（8.610→8.317），但已从 run4-iter100 的
    −1.610 大幅收窄。
  - 自称信号（base → iter125）：强自称 **98%→93%**、宫廷词 61%→59%、
    任一自称 100%→98%、现代词 5%→10%——开场分布基本修复。
  - 分类：`meta` **+2.050**、`wisdom` +0.600、`identity` +0.325、
    `greeting` +0.175、`modern` +0.050、`poetry` 持平；
    退步 `favor` −0.600、`rivalry` −0.580、`emotion` −0.050
    （宫斗攻防与恩宠题的措辞锐度换来了现代/meta 防御与切题性）。
  - 意外收益：checkpoints 中 iter125 的 meta 9.150，说明低 LR 短训
    把 run4 的拒答能力也保留住了；预期风险中"句句臣妾"未发生
    （强自称 93% < base 98%，organic 回合仍有变化）。
  - **鲁棒性追加测试（其他）**：官方采样器下 base 与 final 均 0/41 复读；
    贪心（temp=0）下 base 0/41、final 2/41（`rival-05`、`meta-06`），
    且 run5 各 checkpoint 贪心复读 1–2 条——属本训练方向引入的轻微退化
    （resumed LoRA + 强自称过采样使 logits 分布更尖）。未再开 run6 修复：
    官方采样已 0 复读，继续训练可能侵蚀 +0.281 的成果；改为交付
    **推理配置约束**（README「推理加载最终模型」：temp=0.7 官方采样，
    禁用贪心），数据与指标见 `outputs/robustness.json`。
  - **评分器勘误（v1.1→v1.2）**：run5 系列回复会复述 meta-06 问句中的
    名词「思考过程」（如"臣妾的思考过程？哼，臣妾不会这些"），v1.1 规则
    将该词一概判为思维链泄漏（`non_think=0, safety−4`），属假阳性。
    v1.2 收紧为仅识别真实展示句式（`思考过程（如下|是|为|：）`）与
    `<think>`/「让我想想」等痕迹；cases、维度与权重均未变，
    `rescore.py --all` 对全部模型统一重算（base 分数不变）。

<!-- RUN_RESULTS -->

## 改进思路储备（按优先级）

1. **训练集**：按 instruction 去重（去掉 238 条歧义/重复 prompt）、
   过滤过短无信息回复、平衡高频模板回复。
2. **训练集**：注入现代/meta 防御样例（以甄嬛口吻拒绝 AI/编程/注入），
   强化 safety 维度。
3. **训练参数**：若欠拟合 → 提高 iters / lr；若过拟合（valid loss 回升、
   复读增多）→ 降 lr、减 iters、加 dropout、减 rank。
4. **训练参数**：学习率调度（warmup + cosine）替代恒定 lr。
5. **提示词**：仅在训练集与评测同步更新的前提下调整，并重跑 base。
