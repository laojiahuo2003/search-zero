"""
Turn-level credit assignment for CW-GRPO (Contribution-Weighted GRPO).

Official reference: https://github.com/zsxmwjz/CW-GRPO
  "Enhancing LLM-based Search Agents via Contribution Weighted
   Group Relative Policy Optimization"

This module ports the official mechanism into the search-zero training loop.
It mirrors the upstream implementation (verl/trainer/ppo/ray_trainer.py,
verl/trainer/ppo/core_algos.py, verl/workers/reward_manager/llm_judge.py):

1. Judge every NON-FINAL turn of a trajectory with two binary signals:
     retrieval_reward: the turn's observation brought NEW, relevant
                       information (novelty required: re-retrieving the same
                       facts earns 0, even when relevant).
     thinking_reward:  the turn's THOUGHT is grounded in retrieved evidence
                       and its action aims at genuinely missing information.
     contribution = retrieval_reward * thinking_reward

2. Normalize contributions per trajectory:
     gamma >= 10  -> plain renormalization c / sum(c)
     gamma <  10  -> softmax(gamma * (c - max(c)))   (upstream default gamma=1)

3. Reallocate the trajectory-level GRPO advantage across turns:
     - negative / zero advantage  -> no reallocation (uniform broadcast,
                                     matches upstream: only grpo_adv > 0 is
                                     reallocated)
     - positive advantage         -> non-final turns are scaled by w_t * N
                                     (mean 1: credit-conserving), the final
                                     ANSWER turn keeps the original advantage

Two judge backends, selectable via CREDIT_MODE:
     "rule": deterministic, zero-cost. A NEGATIVE-LIST design: a turn is
             worth zero only when it is clearly useless — the observation
             brought no substantial new information (no-result responses,
             near-duplicate retrieval), or the query is an empty/near
             verbatim repeat. Everything else defaults to 1 and the
             softmax normalization separates the rest. This deliberately
             avoids the previous gold-anchor heuristic, which rewarded
             only observations containing new gold-answer words and
             systematically mis-scored bridge-entity hops in multi-hop
             questions (HotpotQA's first hop rarely contains the answer
             word). A gold word still EXEMPTS a turn from the novelty
             threshold (strong positive signal, no false kill).
     "llm" : the upstream judge prompt adapted to the THOUGHT/ACTION/
             OBSERVATION format, called through the OpenAI-compatible API
             configured in .env (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL).

Configuration (env vars, also settable in .env):
     CREDIT_MODE=none|rule|llm         (default: none -> vanilla GRPO)
     CREDIT_GAMMA=1.0                  softmax inverse temperature (>=10 hard)
     CREDIT_JUDGE_WORKERS=16           parallel LLM judge calls
     CREDIT_RULE_MIN_NEW_WORDS=3       rule judge: min novel words for a
                                       non-gold turn to count as informative
     CREDIT_RULE_QUERY_SIM=0.8         rule judge: Jaccard similarity at or
                                       above which a query counts as a
                                       repeat (rephrases stay useful)
     CREDIT_FALLBACK_UNIFORM=1         uniform weights when all contributions
                                       are 0 (upstream's softmax path behaves
                                       this way; the hard path zeroes them)
     CREDIT_JUDGE_ONLY_POSITIVE_ADV=1  skip judging trajectories whose group
                                       advantage is <= 0 — they are never
                                       reallocated, so judging them is wasted
                                       work (upstream approximates this with
                                       answer_reward > 0; here the exact
                                       group advantage is already known)

Deliberate divergences from upstream (CW-GRPO), audited 2026-09:

1. Advantage source. Upstream feeds answer_reward (binary EM) into
   compute_grpo_outcome_advantage, so only answer correctness shapes the
   group-normalized advantage. search-zero's advantage is (format + accuracy)
   group-normalized — format is a first-class training signal here (the
   ReAct format is what took EM from 3% to 12.5%), so dropping it would
   silently remove that pressure. We keep search-zero's reward design and
   only reallocate the already-computed advantage; the reallocation math
   itself does not depend on which signal produced it.

2. Judge scope. Upstream judges EVERY assistant round including the final
   answer round, then discards the final round's verdict when normalizing
   (its contribution is masked out). We skip the final round entirely —
   same result, one fewer judge call per trajectory.

3. Judge skip condition. Upstream skips judging when answer_reward == 0
   (wrong answer). With search-zero's continuous accuracy reward that test
   is ill-defined, and the true condition is the one the reallocation
   itself uses: group_adv <= 0 trajectories are never reallocated, so we
   skip judging exactly those (CREDIT_JUDGE_ONLY_POSITIVE_ADV). Strictly
   fewer wasted judge calls than upstream for the same math.

4. Response parsing. Upstream accepts only ```json fences; we accept bare
   JSON as well (qwen-class judges frequently omit the fence).
"""
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch
from openai import OpenAI

# Make `app.utils.config` importable regardless of the caller's entry point
# (this module lives in scripts/, the package lives at the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import get_config


# ============================================================
# Configuration
# ============================================================

@dataclass
class CreditConfig:
    mode: str = "none"                 # none | rule | llm
    gamma: float = 1.0                 # softmax inverse temperature
    judge_workers: int = 16            # parallel LLM judge calls
    rule_min_new_words: int = 3        # rule judge novelty threshold
    rule_query_sim_threshold: float = 0.8  # rule judge repeat-query Jaccard
    fallback_uniform: bool = True      # uniform weights when all credits are 0
    judge_only_positive_adv: bool = True  # skip judging adv <= 0 trajectories


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def get_credit_config() -> CreditConfig:
    """Read credit-assignment settings from the environment.

    `get_config()` above already calls load_dotenv(), so `.env` entries are
    visible here as long as app.utils.config was imported first.
    """
    mode = os.getenv("CREDIT_MODE", "none").strip().lower()
    if mode not in ("none", "rule", "llm"):
        raise ValueError(f"CREDIT_MODE must be none|rule|llm, got {mode!r}")
    return CreditConfig(
        mode=mode,
        gamma=float(os.getenv("CREDIT_GAMMA", "1.0")),
        judge_workers=int(os.getenv("CREDIT_JUDGE_WORKERS", "16")),
        rule_min_new_words=int(os.getenv("CREDIT_RULE_MIN_NEW_WORDS", "3")),
        rule_query_sim_threshold=float(os.getenv("CREDIT_RULE_QUERY_SIM", "0.8")),
        fallback_uniform=_env_bool("CREDIT_FALLBACK_UNIFORM", True),
        judge_only_positive_adv=_env_bool("CREDIT_JUDGE_ONLY_POSITIVE_ADV", True),
    )


# ============================================================
# Judge backends
# ============================================================

def _norm_tokens(text: str) -> set:
    """Lowercase, strip punctuation, split into a word set."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return set(text.split())


# LocalWikiSearcher returns these fixed strings for failed lookups; they must
# not be mistaken for informative content (their words would otherwise look
# like "new tokens" the first time they appear).
_NO_RESULT_MARKERS = ("no results found", "empty search query")


def _query_similarity(a: str, b: str) -> float:
    """Jaccard similarity over normalized token sets; identical -> 1.0.

    Token-level, so a rephrased query (synonym swap / added word) lands well
    below the repeat threshold — matching the upstream judge's stance that
    varying a failed query is a useful retrieval attempt.
    """
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def rule_judge_turn(question, gt, turns, idx, cfg=None) -> tuple:
    """Deterministic stand-in for the LLM judge (CREDIT_MODE="rule").

    NEGATIVE-LIST design — a turn scores 0 only when it is clearly useless,
    and defaults to 1 otherwise (the softmax then separates the degrees).
    This keeps false kills rare, because scoring a good hop as 0 re-zeros its
    real contribution, which is worse than letting a mediocre hop keep 1.

    retrieval = 0 when:
        - no observation was produced, or it is a no-result/empty-query
          response from LocalWikiSearcher;
        - the observation brings fewer than cfg.rule_min_new_words novel
          words relative to all previous observations (near-duplicate
          retrieval). A NEW GOLD WORD exempts the turn from the threshold:
          it is the strongest positive signal a rule can see, so even a
          single novel gold word counts as informative.

    thinking = 0 when:
        - the query is empty;
        - the query is a near-verbatim repeat of an earlier query
          (Jaccard >= cfg.rule_query_sim_threshold). Rephrases stay useful.

    `question` is accepted for signature symmetry with the LLM judge but is
    not used: rule-level query/evidence assessment is intentionally
    conservative about what it can reliably know.
    """
    cfg = cfg or CreditConfig()
    turn = turns[idx]
    obs = (turn.get("observation") or "").strip()
    query = (turn.get("query") or "").strip()
    obs_tokens = _norm_tokens(obs)

    # ---- retrieval: substantial novel information (or a new gold word) ----
    retrieval = 0
    if obs and not any(m in obs.lower() for m in _NO_RESULT_MARKERS):
        prev_tokens = set()
        for t in turns[:idx]:
            prev_tokens |= _norm_tokens(t.get("observation") or "")
        new_tokens = obs_tokens - prev_tokens
        gold_new = _norm_tokens(gt) & new_tokens
        if len(new_tokens) >= cfg.rule_min_new_words or gold_new:
            retrieval = 1

    # ---- thinking: non-empty, non-repeat query ----
    thinking = 1
    if not query:
        thinking = 0
    else:
        for t in turns[:idx]:
            prev_q = (t.get("query") or "").strip()
            if prev_q and _query_similarity(query, prev_q) >= cfg.rule_query_sim_threshold:
                thinking = 0
                break

    return retrieval, thinking


def build_trace_text(question: str, turns: list, idx: int) -> str:
    """Render the trajectory prefix up to turn `idx` in the judge's format.

    Mirrors upstream `_trace_to_text`: alternating Agent / Information
    messages; the last action is the one being judged.
    """
    parts = [f"Question: {question}"]
    for t in range(idx + 1):
        parts.append(f"Agent: {turns[t].get('text', '')}")
        obs = turns[t].get("observation")
        if obs:
            parts.append(f"Information: {obs}")
    return "\n\n".join(parts)


def parse_judge_response(content: str):
    """Parse the judge's JSON verdict into (retrieval_reward, thinking_reward).

    Accepts both fenced (```json ... ```) and bare JSON. Returns None when the
    response cannot be parsed so the caller can retry.
    """
    if not content:
        return None
    m = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
    if m:
        content = m.group(1).strip()
    try:
        data = json.loads(content)
    except Exception:
        print(f"[credit-judge] unparseable response: {content[:200]!r}")
        return None
    try:
        ret = int(data["retrieval_reward"])
        thk = int(data["thinking_reward"])
    except (KeyError, TypeError, ValueError):
        print(f"[credit-judge] missing keys in response: {content[:200]!r}")
        return None
    return (1 if ret > 0 else 0, 1 if thk > 0 else 0)


MAX_JUDGE_RETRIES = 3


class LLMTurnJudge:
    """LLM-as-judge for per-turn contribution scoring (CREDIT_MODE="llm").

    Uses the OpenAI-compatible client configured in .env. Results are cached
    per (question, prefix of (query, observation) pairs): the retrieval
    environment is deterministic, so the same turn prefix always receives the
    same verdict, and rollouts frequently repeat queries across steps.
    """

    def __init__(self, api_key=None, base_url=None, model=None, cache_size=10000):
        cfg = get_config()
        self.client = OpenAI(
            api_key=api_key or cfg.llm_api_key,
            base_url=base_url or cfg.llm_base_url,
        )
        self.model = model or cfg.llm_model
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ---- cache ----
    def _cache_get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key, value):
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(question, turns, idx):
        prefix = tuple(
            (t.get("query") or "", t.get("observation") or "")
            for t in turns[: idx + 1]
        )
        return (question, prefix)

    # ---- single judge call ----
    def judge_turn(self, question: str, turns: list, idx: int) -> tuple:
        """Score the action of turn `idx`. Returns (retrieval, thinking)."""
        key = self._cache_key(question, turns, idx)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        trace = build_trace_text(question, turns, idx)
        result = self._call_with_retry(trace)
        self._cache_put(key, result)
        return result

    def _call_with_retry(self, trace: str) -> tuple:
        for attempt in range(MAX_JUDGE_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": f"Here is the trace:\n{trace}"},
                    ],
                    temperature=0.0,
                )
                parsed = parse_judge_response(response.choices[0].message.content)
                if parsed is not None:
                    return parsed
            except Exception as e:
                print(f"[credit-judge] attempt {attempt + 1} failed: {e}")
        return (0, 0)

    # ---- batch judge ----
    def judge_many(self, jobs: list, workers: int = 16) -> list:
        """Judge a batch of (question, turns, idx) jobs in parallel.

        Returns a list of (retrieval, thinking) aligned with `jobs`; a job
        that fails every retry yields (0, 0), matching upstream's fallback.
        """
        results = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.judge_turn, q, turns, idx): i
                for i, (q, turns, idx) in enumerate(jobs)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"[credit-judge] job {i} failed: {e}")
                    results[i] = (0, 0)
        return results


# The upstream judge prompt (verl/workers/reward_manager/llm_judge.py),
# adapted to the THOUGHT / ACTION / OBSERVATION format used by search-zero.
# Scoring criteria and analysis steps are kept verbatim in substance.
JUDGE_PROMPT = """\
You are an assistant that evaluates the validity of the agent's reasoning in a question-answering task.

You will be given a partial trace consisting of alternating messages between the agent and the information source. Your task is to judge **only the latest action** in the trace, based on the agent's latest THOUGHT section and all previously retrieved information.

### Evaluation Criteria

#### Retrieval Reward
For the latest action, give a retrieval_reward of **1** if and only if:

1. **Relevance:**
   The retrieved information is genuinely relevant to the main question and is likely to be helpful in answering it.

2. **Novelty:**
   The retrieved information should offer new, useful content for answering the question that was not already obtained in previous rounds. If the same information (or its substance) was retrieved in previous rounds, do **not** assign a retrieval_reward, even if it is relevant.


#### Thinking Reward
For the latest action, give a thinking_reward of **1** if and only if:

1. **Reasoning Support:**
   The reasoning in the latest THOUGHT section is logically grounded in the previously retrieved documents. The agent's claims, assumptions, and deductions must be supported by the information already obtained.

2. **Action Usefulness:**
   - If the action is a **search**, the proposed retrieval query(s) must aim to obtain missing information that is necessary or beneficial for producing a correct answer.
   - If the action is an **answer**, the reasoning must align with and be properly supported by the retrieved information.
   - When assigning the thinking reward, you must only focus on the content and quality of the agent's latest THOUGHT section; do NOT consider the relevance of the retrieved documents or whether the final answer matches the ground truth.

If **either** the reasoning is unsupported **or** the action is not helpful toward answering the question, assign a score of **0**.

### Additional Notes

- Your judgment must rely **only** on the conversation history and retrieved documents; do not use outside knowledge.
- The knowledge base (KB) used for retrieval was last updated in 2018.
- If multiple documents share the same title, treat them as parts of the same entry. But if the documents have similar but not the same title, treat them as separate and independent.
- The KB may contain documents with names similar to the target entity but that are actually irrelevant.
  If the agent incorrectly treats such similar-but-unrelated documents as relevant and bases reasoning on them, you **must** assign 0.
- Be objective and consistent.

### Analysis Steps
Before you assign the rewards, you should first analyze the latest action in detail. Please follow these steps:

1. Extract the factual claims, reasoning logic, search intent, and assumptions made in the latest action's THOUGHT section.
2. For each factual claim, check whether it is supported by previously retrieved passages or is a matter of common sense.
3. Analyze whether the reasoning logic is rigorous, and whether the search intent aligns with the reasoning and constitutes information still needed to answer the question. **Attention**:
    - If the agent attempts to retrieve information that was previously searched for but not successfully obtained—by rephrasing, using synonyms, or otherwise varying the query—and if that information would be helpful for answering the question, you should consider this retrieval attempt useful.
    - If the agent makes an assumption in place of unavailable information, and the assumption is logically justified, do not penalize the agent for this.
4. Extract information from the retrieved documents that may be relevant to the main question. For statements similar to the question, analyze carefully whether they are truly relevant or only superficially similar but unrelated. If no relevant information is present, skip step 5 and assign a retrieval_reward of 0.
5. For each relevant information, check whether it was already retrieved in previous rounds and, if so, in which round. Finally, give the retrieval_reward based on whether genuinely new relevant information was retrieved in this turn.

Please conduct your analysis in the order above and justify your scoring.

### Format

Input format: a partial multi-round conversation between the agent and the information source. Example:
```
Question: the question to answer
Agent: THOUGHT: ...
ACTION: SEARCH: <the first search query>
Information: the information retrieved by the first search
Agent: THOUGHT: ...
ACTION: SEARCH: <the second search query>
Information: the information retrieved by the second search
...
Agent (the last action): THOUGHT: ...
ACTION: ANSWER: <the final answer>
```
You should **only** evaluate the **last** action.

Return format: a JSON object, wrapped in ```json and ```. Example:
```json
  {"analysis": "your detailed analysis of the latest action", "thinking_reward": 0/1, "retrieval_reward": 0/1}
```\
"""


# ============================================================
# Contribution weights and advantage reallocation
# ============================================================

def normalize_contributions(contribs: list, gamma: float) -> list:
    """Normalize per-turn contributions inside one trajectory.

    Mirrors upstream ray_trainer.py:
      gamma >= 10 -> c / sum(c)            (treat large gamma as infinity)
      otherwise   -> softmax(gamma * (c - max(c)))
    """
    c = torch.tensor(contribs, dtype=torch.float32)
    if gamma >= 10:
        w = c / torch.clamp(c.sum(), min=1e-6)
    else:
        w = torch.softmax(gamma * (c - c.max()), dim=-1)
    return w.tolist()


def build_token_advantages(turns: list, group_adv: float, cfg: CreditConfig):
    """Reallocate the trajectory-level GRPO advantage across turns.

    Port of upstream compute_cw_grpo_advantage:
      - the final (ANSWER) turn keeps the original advantage;
      - non-final turns are scaled by w_t * N, where w_t is the
        trajectory-normalized contribution and N the number of non-final
        turns (mean 1, i.e. credit-conserving);
      - group_adv <= 0 -> no reallocation (uniform broadcast);
      - all-zero contributions -> uniform broadcast when
        cfg.fallback_uniform (upstream's softmax path does this implicitly),
        otherwise non-final turns receive zero advantage.

    `turns` elements carry 'gen_ids' and, for judged turns, a 'credit' float.
    The returned tensor has the same length and order as the concatenated
    per-turn logprobs used by the GRPO loss.
    """
    n_turns = len(turns)
    total = sum(len(t["gen_ids"]) for t in turns)
    base = torch.full((total,), float(group_adv), dtype=torch.float32)
    if group_adv <= 0 or n_turns < 2:
        return base

    n_credit = n_turns - 1
    contribs = [float(t.get("credit", 0.0)) for t in turns[:-1]]

    if sum(contribs) <= 0:
        if cfg.fallback_uniform:
            return base
        weights = [0.0] * n_credit
    else:
        weights = normalize_contributions(contribs, cfg.gamma)

    # Mean 1 over the non-final turns: keeps the trajectory's total credit
    # unchanged, so GRPO's advantage scale is not polluted.
    weights = [w * n_credit for w in weights]

    adv_parts = []
    for i, t in enumerate(turns):
        length = len(t["gen_ids"])
        if i < n_credit:
            adv_parts.append(
                torch.full((length,), weights[i] * float(group_adv), dtype=torch.float32)
            )
        else:
            adv_parts.append(
                torch.full((length,), float(group_adv), dtype=torch.float32)
            )
    return torch.cat(adv_parts)
