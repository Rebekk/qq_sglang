"""
DFlash Speculative Decoding Demo
=================================
DFlash 是 SGLang 的投机解码算法：
  - 目标模型 (target)：完整的大模型，负责验证 draft token
  - 草稿模型 (draft) ：轻量的 DFlash 模型，复用目标模型的 hidden states 快速预测多个 token
  - 每步 decode 先 draft N 个 token，再由 target 一次性验证，实现加速

使用方式
--------
1. 服务端模式（推荐）：
   先启动服务器，再运行 demo 客户端

   # 终端 1：启动服务器
   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.1-8B-Instruct \
       --speculative-algorithm DFLASH \
       --speculative-draft-model-path z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
       --attention-backend flashinfer \
       --page-size 1 \
       --port 30000

   # 终端 2：运行 demo
   python demo_dflash_speculative.py --mode client

2. 内置引擎模式（自动管理服务器生命周期）：
   python demo_dflash_speculative.py --mode engine
"""

import argparse
import time


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SERVER_URL = "http://localhost:30000"

PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms:",
    "Write a short poem about the ocean:",
    "What is the difference between supervised and unsupervised learning?",
]


# ─────────────────────────────────────────────
# 模式 1：客户端模式（服务器已在外部启动）
# ─────────────────────────────────────────────
def run_client_mode(server_url=SERVER_URL):
    import openai

    client = openai.Client(base_url=f"{server_url}/v1", api_key="EMPTY")

    print("=" * 60)
    print("DFlash Speculative Decoding Demo — Client Mode")
    print(f"Server: {server_url}")
    print("=" * 60)

    # ---- Completions API ----
    print("\n[1] Completions API (greedy, temperature=0)")
    print("-" * 40)
    for prompt in PROMPTS[:2]:
        t0 = time.perf_counter()
        resp = client.completions.create(
            model=TARGET_MODEL,
            prompt=prompt,
            max_tokens=64,
            temperature=0,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].text
        tokens = resp.usage.completion_tokens
        print(f"Prompt : {prompt!r}")
        print(f"Output : {text!r}")
        print(f"Tokens : {tokens}  Time: {elapsed:.2f}s  ({tokens/elapsed:.1f} tok/s)")
        print()

    # ---- Chat Completions API ----
    print("\n[2] Chat Completions API (greedy)")
    print("-" * 40)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is speculative decoding and how does it speed up inference?"},
    ]
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=TARGET_MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0,
    )
    elapsed = time.perf_counter() - t0
    content = resp.choices[0].message.content
    tokens = resp.usage.completion_tokens
    print(f"Assistant: {content}")
    print(f"\nTokens: {tokens}  Time: {elapsed:.2f}s  ({tokens/elapsed:.1f} tok/s)")

    # ---- Streaming ----
    print("\n[3] Streaming output")
    print("-" * 40)
    prompt = "List 5 benefits of speculative decoding:"
    print(f"Prompt: {prompt!r}")
    print("Output: ", end="", flush=True)
    t0 = time.perf_counter()
    stream = client.completions.create(
        model=TARGET_MODEL,
        prompt=prompt,
        max_tokens=128,
        temperature=0,
        stream=True,
    )
    total_tokens = 0
    for chunk in stream:
        delta = chunk.choices[0].text
        if delta:
            print(delta, end="", flush=True)
            total_tokens += 1
    elapsed = time.perf_counter() - t0
    print(f"\n\nTokens: ~{total_tokens}  Time: {elapsed:.2f}s")

    # ---- Batched requests (测试并发加速) ----
    print("\n[4] Batched requests")
    print("-" * 40)
    t0 = time.perf_counter()
    responses = []
    for prompt in PROMPTS:
        r = client.completions.create(
            model=TARGET_MODEL,
            prompt=prompt,
            max_tokens=32,
            temperature=0,
        )
        responses.append(r)
    elapsed = time.perf_counter() - t0
    total_tok = sum(r.usage.completion_tokens for r in responses)
    print(f"Completed {len(PROMPTS)} requests, {total_tok} tokens in {elapsed:.2f}s")
    print(f"Average throughput: {total_tok/elapsed:.1f} tok/s")

    print("\n✓ Demo completed.")


# ─────────────────────────────────────────────
# 模式 2：引擎模式（在进程内启动 SGLang Engine）
# ─────────────────────────────────────────────
def run_engine_mode():
    import sglang as sgl

    print("=" * 60)
    print("DFlash Speculative Decoding Demo — Engine Mode")
    print("=" * 60)
    print(f"Target model : {TARGET_MODEL}")
    print(f"Draft model  : {DRAFT_MODEL}")
    print("Initializing engine (this may take a few minutes)…")

    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_MODEL,
        attention_backend="flashinfer",
        page_size=1,
    )

    sampling_params = {"temperature": 0, "max_new_tokens": 64}

    print("\n[1] Single request")
    print("-" * 40)
    prompt = PROMPTS[0]
    t0 = time.perf_counter()
    out = engine.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - t0
    text = out["text"]
    print(f"Prompt : {prompt!r}")
    print(f"Output : {text!r}")
    print(f"Time   : {elapsed:.2f}s")

    print("\n[2] Batch generation")
    print("-" * 40)
    t0 = time.perf_counter()
    outputs = engine.generate(PROMPTS, sampling_params)
    elapsed = time.perf_counter() - t0
    total_tok = sum(o["meta_info"]["completion_tokens"] for o in outputs
                    if "meta_info" in o and "completion_tokens" in o["meta_info"])
    for prompt, out in zip(PROMPTS, outputs):
        p_short = repr(prompt)[:50]
        t_short = repr(out['text'])[:60]
        print(f"  {p_short} → {t_short}")
    print(f"\nTotal time: {elapsed:.2f}s")

    engine.shutdown()
    print("\n✓ Demo completed.")


# ─────────────────────────────────────────────
# 模式 3：快速单元测试（不依赖真实模型）
# ─────────────────────────────────────────────
def run_unit_test():
    """
    演示 DFlash 核心算法逻辑（verify 步骤），不需要 GPU 或真实模型。
    """
    import torch
    from dataclasses import dataclass
    from typing import List

    print("=" * 60)
    print("DFlash Core Algorithm — Unit Test (CPU)")
    print("=" * 60)

    # ── 模拟 compute_dflash_accept_len_and_bonus ──────────────────
    def compute_accept_len_and_bonus(
        candidates: torch.Tensor,   # [bs, draft_token_num]
        target_predict: torch.Tensor,  # [bs, draft_token_num]
    ):
        """
        DFlash 验证：逐位比较 draft token 与 target 预测。
        从位置 0 开始，找到首个不匹配位置，接受前面所有 token，
        并用 target 在该位置的预测作为 bonus token。
        """
        bs, n = candidates.shape
        # candidates[:, 0] 是当前已验证 token（不计入 accept_len），
        # 验证从 candidates[:, 1:] 开始对比 target_predict[:, :-1]
        draft_tokens = candidates[:, 1:]       # [bs, n-1]
        target_tokens = target_predict[:, :-1]  # [bs, n-1]  (verify positions)
        bonus_tokens = target_predict[:, -1]    # [bs]        (unconditional bonus)

        match = (draft_tokens == target_tokens)  # [bs, n-1]

        # 找首个不匹配位置（accept_len = 该位置之前的匹配数）
        first_mismatch = (n - 1) * torch.ones(bs, dtype=torch.long)
        for pos in range(n - 1):
            mask = (~match[:, pos]) & (first_mismatch == (n - 1))
            first_mismatch[mask] = pos

        accept_len = first_mismatch  # 接受的草稿 token 数量

        # bonus token：来自 target 在首个不匹配位置的预测
        bonus = torch.gather(target_predict, 1, first_mismatch.unsqueeze(1)).squeeze(1)

        return accept_len, bonus

    # ── 模拟一次验证迭代 ─────────────────────────────────────────────
    torch.manual_seed(42)
    bs = 3           # batch size
    draft_n = 5      # draft tokens per step（含当前已验证 token）
    vocab_size = 32000

    # 当前已验证 token（每个 request 的"起点"）
    verified_id = torch.randint(0, vocab_size, (bs,))

    # DFlash draft 模型生成的候选序列（首个是 verified_id）
    candidates = torch.cat(
        [verified_id.unsqueeze(1), torch.randint(0, vocab_size, (bs, draft_n - 1))],
        dim=1,
    )

    # Target 模型对每个位置的 argmax 预测（真实情况下来自 forward pass logits）
    target_predict = torch.randint(0, vocab_size, (bs, draft_n))

    # 手动让 request 0 前 2 个 draft token 完全匹配（模拟高 accept rate）
    target_predict[0, :2] = candidates[0, 1:3]

    print(f"\nBatch size       : {bs}")
    print(f"Draft tokens/step: {draft_n - 1} (excluding current token)")
    print(f"\nCandidates (draft):\n{candidates}")
    print(f"\nTarget predictions:\n{target_predict}")

    accept_len, bonus = compute_accept_len_and_bonus(candidates, target_predict)

    print(f"\n── Verification Results ──────────────────────────")
    for i in range(bs):
        acc = int(accept_len[i].item())
        bon = int(bonus[i].item())
        committed = candidates[i, 1 : 1 + acc].tolist() + [bon]
        print(
            f"  Request {i}: accept_len={acc}  bonus={bon}"
            f"  committed_tokens={committed}"
        )

    mean_acc = accept_len.float().mean().item()
    print(f"\nMean accept length: {mean_acc:.2f} / {draft_n - 1}")
    print(
        f"Effective speedup (theoretical): {1 + mean_acc:.2f}x "
        f"(vs 1 token/step baseline)"
    )

    # ── DFlash 注意力掩码构造演示 ────────────────────────────────────
    print("\n── DFlash Verify Attention Mask (Request 0) ──────")
    prefix_len = 8
    q_len = draft_n
    kv_len = prefix_len + q_len

    q_idx = torch.arange(q_len).unsqueeze(1)       # [q, 1]
    k_idx = torch.arange(kv_len).unsqueeze(0)      # [1, kv]
    # 因果掩码：query 可以看到全部 prefix 以及当前位置及之前的 draft token
    mask = k_idx <= (prefix_len + q_idx)           # [q, kv]

    print(f"  prefix_len={prefix_len}  draft_token_num={q_len}  kv_len={kv_len}")
    print(f"  Mask shape: {mask.shape}")
    print("  (rows=query positions, cols=key positions, True=allowed)")
    print(mask.int().numpy())

    print("\n✓ Unit test completed.")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DFlash Speculative Decoding Demo")
    parser.add_argument(
        "--mode",
        choices=["client", "engine", "unit"],
        default="unit",
        help=(
            "client : 连接已运行的 SGLang 服务器 (需先手动启动)\n"
            "engine : 在进程内启动 SGLang Engine (需要 GPU + 模型权重)\n"
            "unit   : 仅运行核心算法单元测试，无需 GPU 或模型 (默认)"
        ),
    )
    parser.add_argument("--url", default=SERVER_URL, help="Server URL (client mode)")
    args = parser.parse_args()

    if args.mode == "client":
        run_client_mode(server_url=args.url)
    elif args.mode == "engine":
        run_engine_mode()
    else:
        run_unit_test()


if __name__ == "__main__":
    main()
