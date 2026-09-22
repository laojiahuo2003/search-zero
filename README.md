# Search-Zero

**基于 GRPO 强化学习训练 7B 模型学会多轮搜索推理**

从零实现 Search-R1 训练管线：搭建 LangGraph ReAct Agent 作为推理骨架，生成 SFT 轨迹数据，再通过 GRPO（Group Relative Policy Optimization）强化学习训练 Qwen2.5-7B 在多轮推理中主动搜索 Wikipedia 并整合信息作答。

在 HotpotQA 多跳推理任务上，5 epoch 训练后 **EM 从 3.0% → 12.5%（+4x），Contains 从 27.0% → 42.3%。**

---

## 架构

```
┌────────────────────────────────────────────────────────────────────┐
│                       Search-Zero                                   │
│                                                                      │
│  ┌────────────────── Training Pipeline ──────────────────────────┐  │
│  │                                                                  │  │
│  │  HotpotQA ──▶ SFT Data Gen ──▶ LoRA SFT ──▶ GRPO RL Training   │  │
│  │   (QA pairs)   (Agent traces)   (Qwen2.5-7B)  (reward-driven)  │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────── Inference Engine ────────────────────────────┐  │
│  │                                                                  │  │
│  │   Question → Query Rewrite → ReAct Loop → Retrieve → Answer     │  │
│  │                  (LangGraph)     (DDG/Tavily)  (FAISS+Rerank)   │  │
│  │                                                                  │  │
│  │   • 最多 5 轮 ReAct 推理                                          │  │
│  │   • Query 自动分解 + 迭代优化                                      │  │
│  │   • BGE 向量检索 + Cross-Encoder 重排序                           │  │
│  │   • 完整推理 Trace 保留                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────── Evaluation ─────────────────────────────────┐  │
│  │  Flask HTTP Proxy + SSH Tunnel → 云端 GPU 实时访问本地 Wikipedia │  │
│  │  HotpotQA: EM / Contains / F1 全自动打分                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 核心设计

### 训练管线（重点）

项目分三个阶段，每个阶段解决不同的工程挑战：

#### Phase 1: 生成 SFT 数据

用 GPT-4o-mini 驱动的 ReAct Agent 在 HotpotQA 问题上运行，生成搜索推理轨迹（Thought → Search → Observation → … → Answer）。这些轨迹就是训练数据——模型要学习"在什么情况下该搜什么、搜完怎么整合信息"。

```
python scripts/generate_sft_data.py data/hotpotqa_dev.json 1000 -w 16
```

#### Phase 2: LoRA SFT 微调

用 LLaMA-Factory 在 Qwen2.5-7B-Instruct 上做 LoRA 微调，让模型初步学会 ReAct 格式和多轮搜索行为。

```bash
llamafactory-cli train configs/sft_lora.yaml
```

#### Phase 3: GRPO 强化学习 ★

这是项目的核心。用自定义训练循环实现 GRPO：

```
每个训练 step：
  1. 模型对一个问题生成 N=4 条推理轨迹（rollout）
  2. 每条轨迹实际调用 Wikipedia 搜索（真实工具交互）
  3. Reward 函数打分（格式分 + Contains + EM，三层连续 reward）
  4. GRPO Loss：组内相对比较，advantage 驱动策略更新
  5. 解决 enable_input_require_grads 梯度流问题
```

`train_grpo_search.py`（~600 行）实现了完整的 GRPO 训练循环，不需要 RL 框架依赖。

**关键工程决策：**

| 问题 | 方案 | 原因 |
|------|------|------|
| 多轮 token 对齐 | 原始文本拼接 + Qwen2.5 chat markers | 保持 token ID 在生成和 loss 计算间确定性一致 |
| 梯度流断裂 | `enable_input_require_grads` + incremental forward | 多轮 forward 后梯度链不能断 |
| advantage 稀疏 | 三层连续 reward 替代二值判定 | 格式分 + Contains + EM，GRPO 组内差异化 |

### Reward 函数设计

```
Reward = 0.1 × Format(是否正确输出 THOUGHT/ACTION 格式)
       + 0.3 × Contains(答案关键词是否在标准答案中出现)
       + 0.6 × EM(完全匹配)

# 连续 reward 替代二值 0/1，组内 advantage 更平滑
```

### Agent 推理引擎（LangGraph）

ReAct Agent 基于 LangGraph StateGraph 构建，非黑盒封装：

```
Init → Think → Search → Reflect → Think → ... → Answer
         ↑                  │
         └──────────────────┘ (继续搜索)
                            │
                            └──────────────→ Answer (信息充分或达到最大步数)
```

每个节点的行为完全可控——`react_agent.py` 的 `_node_*` 方法清晰定义了状态转换。

---

## 项目结构

```
search-zero/
│
├── app/
│   ├── agent/              # ReAct Agent（LangGraph）
│   │   ├── react_agent.py  # 状态图定义 → 核心推理引擎
│   │   ├── state.py        # AgentState / AgentStep
│   │   └── prompts.py      # ReAct System Prompt
│   │
│   ├── tools/              # 搜索工具
│   │   ├── search.py       # DuckDuckGo + Tavily（双 provider）
│   │   └── base.py         # Document / ToolResult
│   │
│   ├── planner/            # Query 规划
│   │   └── query_rewriter.py  # 问题分解 + 迭代优化
│   │
│   ├── retrieval/          # 向量检索
│   │   ├── embedder.py     # BGE 编码
│   │   └── vector_store.py # FAISS 索引
│   │
│   ├── reranker/           # 重排序
│   │   └── reranker.py     # BGE Cross-Encoder
│   │
│   ├── evaluation/         # 评测
│   │   ├── metrics.py      # EM / Contains / F1
│   │   └── benchmark.py    # Baseline RAG vs Search-R1 对比
│   │
│   ├── api/                # FastAPI 服务
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   └── utils/              # 工具
│       ├── config.py       # 环境变量配置
│       └── llm.py          # LLM 统一接口
│
├── scripts/
│   ├── generate_sft_data.py    # ★ SFT 数据生成（Agent 轨迹采集）
│   ├── filter_sft_data.py      # SFT 数据质量过滤
│   ├── train_sft.py            # SFT 微调启动
│   ├── train_grpo_search.py    # ★ GRPO 训练循环（~600行，核心）
│   ├── build_wiki_index.py     # Wikipedia 本地索引构建
│   ├── wiki_search.py          # Wikipedia 本地搜索
│   ├── wiki_search_server.py   # Wikipedia 搜索服务
│   ├── eval_with_real_wiki.py  # 带 Wikipedia 搜索的评测
│   ├── run_eval.py             # 标准评测
│   ├── download_models.py      # 模型下载
│   └── download_hotpotqa.py    # 数据下载
│
├── configs/
│   ├── grpo.yaml               # GRPO 训练配置
│   ├── sft_lora.yaml           # SFT LoRA 配置
│   └── dataset_info.json       # LLaMA-Factory 数据集注册
│
├── frontend/
│   └── app.py                  # Streamlit Demo
│
├── tests/
│   └── test_agent.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 安装

### 前置条件

- Python 3.11 / 3.12（由 `.python-version` 固定，uv 会自动装）
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- GPU：SFT/GRPO 训练需要 ≥16GB VRAM；纯 ReAct Agent 推理只需 CPU

依赖拆成了 extra，按需安装，不必一次装全：

| 命令 | 装了什么 | 适用场景 |
|------|----------|----------|
| `uv sync` | LangGraph Agent + OpenAI SDK + DuckDuckGo | 只想跑 Agent / 调 API |
| `uv sync --extra retrieval` | + BGE embedding + FAISS | 本地向量检索 |
| `uv sync --extra train` | + torch(CUDA) / transformers / peft / datasets / accelerate / trl | SFT + GRPO 训练 |
| `uv sync --extra wiki` | + `wikipedia` | 训练时调用真实 Wikipedia 搜索 |
| `uv sync --extra server --extra demo` | + FastAPI / uvicorn / Streamlit | 起服务和 Demo |
| `uv sync --all-extras` | 全部 | 完整开发环境 |

> ⚠️ `uv sync --extra X` 会**卸掉**其他没指定的 extra。要保留多个就用
> `uv sync --all-extras`，或把 extra 并列写出：`uv sync --extra train --extra retrieval`。

```bash
# 1. 安装依赖（uv 会自动创建 .venv 并下载锁定版本的 Python）
uv sync --all-extras

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM API Key

# 3. （可选）预下载 embedding 模型
uv run python scripts/download_models.py
```

torch 固定走 `download.pytorch.org/whl/cu126`，其余包走清华源，见
[pyproject.toml](pyproject.toml) 的 `[tool.uv]` 段。国内网络下无需额外配置。

`uv.lock` 已提交，`uv sync` 会精确复现同一套版本。需要 pip 格式的依赖列表：

```bash
uv export --no-hashes -o requirements.txt
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key | — |
| `LLM_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名 | `gpt-4o-mini` |
| `SEARCH_PROVIDER` | 搜索引擎（`duckduckgo` / `tavily`） | `duckduckgo` |
| `AGENT_MAX_STEPS` | 最大 ReAct 轮数 | `5` |

---

## 使用

### 单次搜索推理

```bash
# Streamlit Demo（可视化推理过程）
streamlit run frontend/app.py

# 或 FastAPI 服务
python -m app.api.main
# → http://localhost:8000/docs
```

### API 调用

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "Which programming language was created first: Python or JavaScript?"}'
```

### 完整训练管线

```bash
# Step 1: 下载 HotpotQA 数据
python scripts/download_hotpotqa.py

# Step 2: 构建 Wikipedia 本地搜索索引
python scripts/build_wiki_index.py

# Step 3: 生成 SFT 轨迹数据（用强模型"教"弱模型）
python scripts/generate_sft_data.py data/hotpotqa_dev.json 1000 -w 16

# Step 4: LoRA SFT 微调
python scripts/train_sft.py

# Step 5: GRPO 强化学习训练
python scripts/train_grpo_search.py
```

### 评测

```bash
# 标准评测
python scripts/run_eval.py data/hotpotqa_dev.json 100

# 带 Wikipedia 搜索的评测
python scripts/eval_with_real_wiki.py
```

---

## 训练结果（HotpotQA 多跳推理）

| 阶段 | EM | Contains | 说明 |
|------|-----|----------|------|
| Qwen2.5-7B（基座，zero-shot） | 3.0% | 27.0% | 不会搜索，纯靠参数知识 |
| SFT 后 | 7.8% | 35.1% | 学会了 ReAct 格式，开始主动搜索 |
| GRPO 1 epoch | 10.2% | 38.7% | reward 驱动下搜索行为更精准 |
| GRPO 5 epoch | **12.5%** | **42.3%** | 多轮搜索 + 信息整合能力持续提升 |

*训练配置：Qwen2.5-7B-Instruct, LoRA rank=8, 500 条 HotpotQA 训练样本, A100 80G ×1*

---

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Agent 框架 | **LangGraph** | ReAct 状态机 |
| LLM 接口 | OpenAI SDK（兼容格式） | 推理生成 |
| 搜索 | DuckDuckGo / Tavily | 信息检索 |
| 训练框架 | PyTorch + Transformers + PEFT | GRPO 自定义训练循环 |
| SFT 工具 | LLaMA-Factory | LoRA 微调 |
| 向量检索 | BGE + FAISS | Dense retrieval |
| 重排序 | BGE Cross-Encoder | 精确匹配 |
| API 服务 | FastAPI | REST + SSE 流式 |
| 前端 | Streamlit | 交互式 Demo |

---

## 为什么这个项目值得展示

这个项目的真实价值不在于最终的 12.5% EM（7B 模型 + 有限预算下不可能刷榜），而在于：

1. **完整管线能力** — 从数据生成 → SFT → GRPO RL → 评测，整个闭环自己搭
2. **工程问题解决** — 梯度流、token 对齐、advantage 稀疏，这些都是 RL 训练的"真实世界"问题
3. **对模型行为的判断力** — 知道 GRPO 什么时候有效、reward 怎么设计不退化、多轮推理的计算瓶颈在哪
4. **Harness 视角** — Agent 不是黑盒，每一个节点和状态转换都是可控的

---

## License

MIT
