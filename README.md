# Search-Zero

**从零实现 Search-R1：GRPO 强化学习训练 7B 模型学会多轮搜索推理**

完整自研训练管线：LangGraph ReAct Agent 采集 SFT 轨迹 → LoRA SFT → 自定义 GRPO 训练循环（无任何 RL 框架依赖），让 Qwen2.5-7B 在多轮推理中主动搜索 Wikipedia 并整合信息作答。

HotpotQA 多跳推理（100 题评测集，EM / Contains）：

| 阶段 | EM | Contains | 说明 |
|------|-----|----------|------|
| Qwen2.5-7B 基座（zero-shot） | 3.0% | 27.0% | 不会搜索，纯靠参数知识 |
| SFT 后 | 7.8% | 35.1% | 学会 ReAct 格式，开始主动搜索 |
| GRPO（历史 5 epoch，未优化管线） | 12.5% | 42.3% | A100 80G ×1 |
| **GRPO（优化管线，1 epoch）** | **14.0%** | **45.0%** | 同口径评测，训练时间 ~71 分钟 |

优化管线 = 跨样本打包生成 + 打包 logprob 计算 + 本地确定性检索环境。**单 epoch 达到历史 5 epoch 效果，算力成本压缩 5 倍。**

> ⚠️ **数字口径说明（待重测）**：上表两组 GRPO 数字由旧版评测脚本测得——其 prompt 构造与训练不一致、且为采样解码。评测脚本已修正为与训练逐 token 一致的口径（见[评测](#评测)章节），这两行数字待用新口径重测后刷新。

---

## 快速开始

按顺序执行，需要 GPU 的步骤已标注：

| 步骤 | 命令 | GPU |
|------|------|-----|
| 1. 安装 | `uv sync --all-extras` + `cp .env.example .env` | — |
| 2. 下载模型 | `uv run python scripts/download_models.py` | — |
| 3. 下载数据 | `uv run python scripts/download_hotpotqa.py` | — |
| 4. 建 Wiki 索引 | `uv run python scripts/build_wiki_index.py` | — |
| 5. 采 SFT 轨迹 | `uv run python scripts/generate_sft_data.py data/hotpotqa_dev.json 1000 -w 16` | — |
| 6.（可选）过滤 | `uv run python scripts/filter_sft_data.py -w 16` | — |
| 7. LoRA SFT | `llamafactory-cli train configs/sft_lora.yaml`（LLaMA-Factory 需单独安装） | ✅ |
| 8. GRPO 训练 | `uv run python scripts/train_grpo_search_MI300X.py` | ✅ |
| 9. 评测 | 见下文 | ✅ |

### 安装说明

依赖拆成 extra，按需安装：

| 命令 | 适用场景 |
|------|----------|
| `uv sync` | 只跑 Agent / 调 API |
| `uv sync --extra retrieval` | 本地向量检索（BGE + FAISS） |
| `uv sync --extra train` | SFT + GRPO 训练 |
| `uv sync --extra tracking` | SwanLab 实验记录 |
| `uv sync --extra server --extra demo` | FastAPI 服务 + Streamlit Demo |
| `uv sync --all-extras` | 完整开发环境 |

> ⚠️ `uv sync --extra X` 会卸掉未指定的 extra，多个并列写出或直接 `--all-extras`。

数据/模型/checkpoint 的落点由环境变量 `SEARCH_ZERO_ROOT` 决定（默认仓库根目录），
路径全部由 [app/utils/config.py](app/utils/config.py) 派生，换机器只改这一个变量：

```bash
# .env
SEARCH_ZERO_ROOT=/mnt/workspace   # → {models,data,outputs}/ 子目录
```

LLaMA-Factory（步骤 7）不是本项目依赖，需单独安装；本项目只通过
`configs/sft_lora.yaml` 这个 YAML 契约和它交互，代码里没有 `import llamafactory`。

### 评测

```bash
# 默认本地索引模式：与训练检索环境完全一致（确定性、无外网依赖）
uv run python scripts/eval_with_real_wiki.py \
  --checkpoint outputs/search_r1_grpo_search \
  --eval_data data/hotpotqa_eval_100.json \
  --output eval_results.json

# 小显存 GPU（如 RTX 3080 10GB）加 4-bit 量化
uv run python scripts/eval_with_real_wiki.py \
  --checkpoint outputs/search_r1_grpo_search_0929 \
  --eval_data data/hotpotqa_eval_100.json --load_in_4bit

# 标准对比评测：Baseline RAG vs Search-R1
uv run python scripts/run_eval.py data/hotpotqa_dev.json 100
```

评测的**训评一致性契约**：rollout 直接复用训练脚本的 `GRPO_SYSTEM_PROMPT` /
`make_prompt_ids` / `tokenize_observation` / `extract_answer`，prompt 和
observation 包装与训练逐 token 一致；解码为 greedy（可复现）。

如需联网 Wikipedia 评测：先 `uv run python scripts/wiki_search_server.py`，
再加 `--wiki_mode url --wiki_url http://127.0.0.1:18080/search`。

### 实验记录（SwanLab）

训练脚本自动接 SwanLab，**不装也能跑**——所有调用容错，没配 key 时静默降级到本地 `./swanlog`：

```bash
# .env
SWANLAB_API_KEY=xxxxxxxx
# SWANLAB_MODE=online|local|disabled
```

---

## 训练核心设计（重点）

核心实现：[scripts/train_grpo_search_MI300X.py](scripts/train_grpo_search_MI300X.py)（自定义 GRPO 训练循环，零 RL 框架依赖；[train_grpo_search.py](scripts/train_grpo_search.py) 保留为原始基线）。

```
每个训练 step：
  Phase 1   生成：P×G 条轨迹打包成一次 batched generate()，多轮 ReAct + 真实检索，
            同步计算 old_logprobs（no_grad，存 CPU）
  Phase 2   奖励：format + accuracy 连续奖励 → 组内归一化 advantage
  Phase 2.5 信用分配（可选）：CW-GRPO 逐轮贡献打分，重分配 advantage
  Phase 3   学习：打包重放 new_logprobs（保留梯度）→ PPO-style clip loss，
            内层更新 µ 次（old_logprobs 冻结）
```

### 关键工程决策

| 问题 | 方案 | 收益 |
|------|------|------|
| 检索环境噪声污染 advantage | **本地确定性 Wiki 索引**（`LocalWikiSearcher`，词重叠打分，内存缓存） | query→observation 确定映射，组内 advantage 只反映策略差异；零延迟、可离线、可大规模并发 |
| Launch-bound GPU（ROCm 实测 ~84µs/kernel 固定开销） | **跨样本打包生成**：P×G 条序列左 padding 进一次 `generate()` | epoch 2.3h → 71min |
| logprob 逐条重放调用次数爆炸 | **打包 logprob**：跨 completion 去重 (input_ids, gen_ids)，左 padding 批量前向 | 96 次/微步 → 24 次 |
| 打包前向下逐 completion 顺序 backward 二次反传崩溃 | group 内 loss **求和后单次 backward**（梯度可加，数学等价，峰值显存不变） | 修复潜伏 bug |
| ratio≡1，clip 形同虚设 | **内层多轮更新 µ**：rollout 采一次，Phase 3 冻结 old_logprobs 重放 µ 次 | clip 真正生效，提升样本效率 |
| 熵坍缩 | clip-higher 非对称裁剪（0.2 / 0.28，对齐 DAPO） | 保留探索能力 |
| 结果奖励稀疏、信用分配粗糙 | **CW-GRPO**：逐轮贡献权重重分配 advantage | 见下文 |
| 多轮 token 对齐 | 原始文本拼接 + Qwen2.5 chat markers（硬编码） | 生成与 loss 计算间 token ID 确定性一致 |
| LoRA 梯度流断裂 | `enable_input_require_grads` | 多轮 forward 后梯度链不断 |

### Reward 设计

```
Reward = Format + Accuracy

Format（封顶 1.0）      THOUGHT +0.3 / ACTION +0.3 / ANSWER +0.5
Accuracy                EM 1.0 / gold⊂pred 0.7 / pred⊂gold 0.5 / 词重叠 0.1–0.4
```

连续奖励替代二值 0/1，G=2 小组内也能产生差异化 advantage。

### CW-GRPO 信用分配（可选）

[scripts/credit_assignment.py](scripts/credit_assignment.py)：把轨迹级 advantage 按逐轮贡献重新分配（移植自 CW-GRPO 官方实现，与 IGPO / StepSearch 的信息增益思路同向）：

- 判定非末尾轮次的 retrieval × thinking 两个二值信号，归一化后重分配（credit 守恒，mean(w)=1，不污染 advantage 尺度）
- 两种 judge：**`rule`**（零成本，负面清单设计——只惩罚明确无信息增益的轮次，避免多跳桥接检索被误杀）和 **`llm`**（OpenAI 兼容 API，带 query 级缓存）
- 成本控制：只判 advantage>0 的轨迹、只判含 SEARCH 的轮次、advantage≤0 不重分配

```bash
# .env
CREDIT_MODE=none|rule|llm     # 默认 none = 原生 GRPO
CREDIT_GAMMA=1.0              # >=10 为硬归一化
```

### 内层多轮更新（µ）

单次更新结构下 old/new logprob 由同一组参数算出，ratio≡1，PPO clip 从不生效。设 `GRPO_INNER_UPDATES=3` 后，rollout 数据采一次、Phase 3 重放 3 次，从第 2 次起参数已变、ratio 偏离 1，clip 真正开始约束更新。生成（最贵的阶段）仍只跑一次，每次额外内层只花一遍打包 logprob + backward。

```bash
GRPO_INNER_UPDATES=3 uv run python scripts/train_grpo_search_MI300X.py
```

µ>1 时 SwanLab 额外记录 `ratio_dev`（mean |ratio−1|）与 `clip_frac`（被裁剪 token 比例）——clip 是否干活的直接证据。µ=1（默认）行为与历史版本严格一致。

### 训练超参

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-7B-Instruct + LoRA（SFT checkpoint 热启动） |
| 组大小 G / 每 micro 样本 P / 梯度累积 | 2 / 8 / 2 |
| 最大轮次 / 每轮 token | 3 / 256 |
| lr / clip / KL β / 温度 | 5e-7 / 0.2–0.28 / 0.04 / 0.9 |
| 训练样本 | 500 条 HotpotQA |
| 硬件 | A100 80G（历史）/ AMD MI300X（当前主线，ROCm + SDPA） |

---

## Agent 推理引擎

ReAct Agent 基于 LangGraph StateGraph，非黑盒封装，每个节点可控：

```
Init → Think → Search → Reflect → Think → ... → Answer
```

- 最多 5 轮 ReAct 推理，Query 自动分解 + 迭代优化
- DuckDuckGo / Tavily 双搜索 provider，BGE 向量检索 + Cross-Encoder 重排序
- FastAPI 服务（REST + SSE 流式）+ Streamlit 可视化 Demo

```bash
uv run streamlit run frontend/app.py     # Demo
uv run python -m app.api.main            # API → http://localhost:8000/docs
```

---

## 项目结构

```
search-zero/
├── app/
│   ├── agent/           # ReAct Agent（LangGraph 状态图）
│   ├── tools/           # DuckDuckGo + Tavily 搜索
│   ├── planner/         # Query 分解与改写
│   ├── retrieval/       # BGE + FAISS 向量检索
│   ├── reranker/        # BGE Cross-Encoder 重排序
│   ├── evaluation/      # EM / Contains / F1
│   ├── api/             # FastAPI 服务
│   └── utils/           # 配置（SEARCH_ZERO_ROOT）/ LLM 接口 / SwanLab
├── scripts/
│   ├── train_grpo_search_MI300X.py  # ★ GRPO 训练主线（打包生成/打包 logprob/CW-GRPO/内层更新）
│   ├── train_grpo_search.py         # GRPO 原始基线
│   ├── train_grpo.py                # trl GRPOTrainer 对照路径（非主线）
│   ├── test_adapter.py              # 本地 4-bit 冒烟测试 LoRA 适配器
│   ├── credit_assignment.py         # CW-GRPO 信用分配（rule/llm 双 judge）
│   ├── wiki_search.py               # 本地确定性检索 LocalWikiSearcher
│   ├── build_wiki_index.py          # Wiki 索引构建
│   ├── generate_sft_data.py         # SFT 轨迹采集
│   ├── filter_sft_data.py           # LLM 裁判过滤
│   ├── train_sft.py                 # SFT 启动包装
│   └── eval_with_real_wiki.py       # 评测（本地索引/联网双模式）
├── configs/             # sft_lora.yaml / dataset_info.json
├── frontend/            # Streamlit Demo
├── tests/               # 68 项测试（CPU 假模型，无需 GPU）
└── pyproject.toml       # uv 管理，uv.lock 已提交
```

## 测试

```bash
.venv/bin/python -m pytest tests/ -v    # 68 项，全部 CPU 可跑
```

覆盖：打包生成的行序契约/左 padding/混合结束、打包 logprob 与逐条计算的逐 token 等价性与梯度等价性、CW-GRPO 规则裁判、内层更新（µ=1 与参考实现逐参数一致 / old_logprobs 冻结 / clip 真实触发 / micro 划分不变性）、评测本地索引模式。

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph（ReAct 状态机） |
| 训练 | PyTorch + Transformers + PEFT（自定义 GRPO 循环，无 RL 框架） |
| SFT | LLaMA-Factory（外部 YAML 契约） |
| 检索（训练） | 本地确定性 Wiki 索引 |
| 检索（推理） | DuckDuckGo / Tavily + BGE + FAISS + Cross-Encoder |
| 实验记录 | SwanLab（容错降级） |
| 包管理 | uv（`uv.lock` 精确复现） |

---

## 为什么这个项目值得展示

最终 14% EM 不是重点（7B + 500 样本 + 有限预算不可能刷榜），真正的价值在：

1. **完整管线闭环** — 数据采集 → SFT → GRPO RL → 评测，每个环节自己搭、自己验证
2. **真实的 RL 工程问题** — 二次反传崩溃、token 对齐、advantage 稀疏、ratio≡1 结构性失效，每一个都是教科书不会告诉你的坑
3. **系统性能功底** — 从 profiling 出发定位 launch-bound 瓶颈，打包生成 + 打包 logprob 把 epoch 压缩 5 倍，且数学等价性有测试背书
4. **与前沿工作的独立对齐** — 确定性检索环境（ZeroSearch 同哲学）、信息增益信用分配（IGPO/StepSearch 同向）、clip-higher（DAPO）、内层更新（标准 PPO 内循环），设计判断经得起对照
5. **训练-评测口径一致** — 确定性环境保证 advantage 信号纯净，评测默认复用训练同款检索后端

## License

MIT
