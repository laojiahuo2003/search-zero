"""
验证 per-completion backward 和 sum-then-backward 的梯度等价性。

用一个小模型 + 随机数据模拟 Phase 3 的两种写法，检查参数梯度是否一致。
"""
import torch
import torch.nn as nn

# 模拟一个小模型
class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # (B, L) -> (B, L, V)
        x = self.embed(input_ids)
        return self.proj(x)

def compute_logprobs(model, input_ids, target_ids):
    """模拟 compute_batch_logprobs 的逻辑"""
    logits = model(input_ids)  # (B, L, V)
    # 只取最后几个 token（模拟生成部分）
    L_gen = target_ids.size(1)
    gen_logits = logits[:, -L_gen:, :]  # (B, L_gen, V)

    # gather + logsumexp
    gathered = gen_logits.gather(2, target_ids.unsqueeze(2)).squeeze(2)
    lse = torch.logsumexp(gen_logits.float(), dim=-1)
    return gathered.float() - lse

def grpo_loss_simple(new_lps, old_lps, adv):
    """简化版 GRPO loss"""
    log_ratio = new_lps - old_lps
    return -(torch.exp(log_ratio) * adv).sum()

# ============================================================
# 测试：4 个 completion，每个 3 个 turn
# ============================================================
torch.manual_seed(42)
model = TinyLM()
vocab_size = 100

# 模拟一个 sample 的 4 个 completion，每个 3 个 turn
# 为了简化，每个 turn 用相同的输入长度和生成长度
num_completions = 4
num_turns_per_completion = 3
seq_len = 10
gen_len = 5

# 生成随机输入和目标（模拟 group_turns）
all_input_ids = []
all_target_ids = []
all_old_lps = []
all_advantages = []

for g in range(num_completions):
    comp_inputs = []
    comp_targets = []
    comp_old_lps = []
    for t in range(num_turns_per_completion):
        inp = torch.randint(0, vocab_size, (seq_len,))
        tgt = torch.randint(0, vocab_size, (gen_len,))
        old_lp = torch.randn(gen_len)  # 模拟 old_logprobs
        comp_inputs.append(inp)
        comp_targets.append(tgt)
        comp_old_lps.append(old_lp)

    all_input_ids.append(comp_inputs)
    all_target_ids.append(comp_targets)
    all_old_lps.append(torch.cat(comp_old_lps))  # 拼接成一个向量
    all_advantages.append(torch.randn(1).item())  # 标量 advantage

# ============================================================
# 方法 1：原始方法 — 一次算完所有 completion，sum loss，backward 一次
# ============================================================
model1 = TinyLM()
model1.load_state_dict(model.state_dict())
model1.zero_grad()

total_loss1 = 0.0
for g in range(num_completions):
    # 模拟 batched_compute_turn_logprobs：把这个 completion 的所有 turn 拼成一个 batch
    # 为简化，我们直接逐 turn 算再 cat（实际代码里是 packed forward）
    turn_new_lps = []
    for t in range(num_turns_per_completion):
        inp = all_input_ids[g][t].unsqueeze(0)  # (1, seq_len)
        tgt = all_target_ids[g][t].unsqueeze(0)  # (1, gen_len)
        lps = compute_logprobs(model1, inp, tgt).squeeze(0)  # (gen_len,)
        turn_new_lps.append(lps)

    new_lps = torch.cat(turn_new_lps)
    old_lps = all_old_lps[g]
    adv = all_advantages[g]

    loss = grpo_loss_simple(new_lps, old_lps, adv)
    total_loss1 = total_loss1 + loss

# 一次 backward
total_loss1.backward()

# 收集梯度
grads1 = {name: p.grad.clone() for name, p in model1.named_parameters() if p.grad is not None}

# ============================================================
# 方法 2：新方法 — 逐个 completion forward + backward
# ============================================================
model2 = TinyLM()
model2.load_state_dict(model.state_dict())
model2.zero_grad()

total_loss2 = 0.0
for g in range(num_completions):
    turn_new_lps = []
    for t in range(num_turns_per_completion):
        inp = all_input_ids[g][t].unsqueeze(0)
        tgt = all_target_ids[g][t].unsqueeze(0)
        lps = compute_logprobs(model2, inp, tgt).squeeze(0)
        turn_new_lps.append(lps)

    new_lps = torch.cat(turn_new_lps)
    old_lps = all_old_lps[g]
    adv = all_advantages[g]

    loss = grpo_loss_simple(new_lps, old_lps, adv)
    total_loss2 += loss.item()

    # 立即 backward
    loss.backward()

# 收集梯度
grads2 = {name: p.grad.clone() for name, p in model2.named_parameters() if p.grad is not None}

# ============================================================
# 比较
# ============================================================
print("=" * 60)
print("梯度等价性验证")
print("=" * 60)
print(f"方法 1 total_loss: {total_loss1.item():.6f}")
print(f"方法 2 total_loss: {total_loss2:.6f}")
print(f"Loss 差异: {abs(total_loss1.item() - total_loss2):.2e}")
print()

all_close = True
for name in grads1:
    g1 = grads1[name]
    g2 = grads2[name]

    # 相对误差
    abs_diff = (g1 - g2).abs()
    rel_diff = abs_diff / (g1.abs() + 1e-8)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    # 用 torch.allclose 的默认容差：rtol=1e-5, atol=1e-8
    close = torch.allclose(g1, g2, rtol=1e-5, atol=1e-8)

    print(f"{name:20s}  max_abs_diff={max_abs:.2e}  max_rel_diff={max_rel:.2e}  allclose={close}")

    if not close:
        all_close = False

print()
if all_close:
    print("✓ 所有参数的梯度在数值容差内完全一致")
    print("✓ 两种方法数学等价，可以安全替换")
else:
    print("✗ 存在梯度差异超出容差")
    print("  （这通常是浮点累加顺序导致的，不影响训练）")
