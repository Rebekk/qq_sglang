"""
SGLang EAGLE Speculative Decoding + Qwen 使用 Demo
====================================================

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是一种
基于特征级草稿（feature-level draft）的推测解码方法。与 token-level draft 不同，
EAGLE 的草稿模型接收 target 模型最后一层的 hidden state，因此草稿质量更高，
接受率通常在 80-90%+，可实现 2-3× 的吞吐提升。

本 demo 演示如何在 SGLang 中使用 EAGLE + Qwen2.5-7B-Instruct，包括：
  - 服务器模式（server + client 分离，适合生产部署）
  - 离线引擎模式（Engine API，适合快速实验）
  - 基线对比（普通解码 vs EAGLE 解码，展示加速比）

模型推荐
--------
  Target : Qwen/Qwen2.5-7B-Instruct
  Draft  : leptonai/EAGLE-Qwen2.5-7B-Instruct

  更大的模型对（预期加速比更高）：
  Target : Qwen/Qwen2.5-72B-Instruct
  Draft  : yuhuili/EAGLE-Qwen2.5-72B-Instruct

EAGLE 核心参数说明
------------------
  speculative_algorithm        : "EAGLE"（固定）
  speculative_draft_model_path : draft 模型路径（必须与 target 匹配）
  speculative_num_steps        : 每次 draft 的前向步数，推荐 3-5
                                  越大候选树越深，但不一定更快（受接受率影响）
  speculative_eagle_topk       : 每步保留 top-k 候选，推荐 4-8
                                  topk↑ → 候选宽度↑ → 接受率↑，但内存↑
  speculative_num_draft_tokens : 从候选树中最终选出的草稿 token 数
                                  通常设为 speculative_num_steps × speculative_eagle_topk 的子集

使用方法（服务器模式）
----------------------
Step 1: 在另一个终端启动服务器

  # 带 EAGLE 的 Qwen2.5-7B
  python -m sglang.launch_server \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --speculative-algorithm EAGLE \\
      --speculative-draft-model-path leptonai/EAGLE-Qwen2.5-7B-Instruct \\
      --speculative-num-steps 3 \\
      --speculative-eagle-topk 4 \\
      --speculative-num-draft-tokens 16 \\
      --cuda-graph-max-bs 8 \\
      --port 30000

  # 不带 EAGLE 的基线（用于对比）
  python -m sglang.launch_server \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --port 30001

Step 2: 运行本 demo

  # 只测 EAGLE
  python eagle_qwen_demo.py --port 30000

  # 和基线对比
  python eagle_qwen_demo.py --port 30000 --baseline-port 30001 --compare

  # 使用离线 Engine 模式（无需启动服务器）
  python eagle_qwen_demo.py --offline

  # 使用 chat 模板（OpenAI /v1/chat/completions）
  python eagle_qwen_demo.py --port 30000 --chat
"""

import argparse
import json
import time
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 推荐模型对（可通过 --model-path / --draft-model-path 覆盖）
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = "leptonai/EAGLE-Qwen2.5-7B-Instruct"

# EAGLE 推荐参数
EAGLE_NUM_STEPS = 3          # draft 步数
EAGLE_TOPK = 4               # 每步 top-k
EAGLE_NUM_DRAFT_TOKENS = 16  # 最终候选 token 数

DEMO_PROMPTS = [
    "请简要介绍量子计算的基本原理，并举一个实际应用的例子。",
    "写一段 Python 代码，实现归并排序算法，并分析其时间和空间复杂度。",
    "Explain the difference between supervised learning, unsupervised learning, and reinforcement learning.",
    "解释 Transformer 架构中 Self-Attention 机制的工作原理，用公式说明。",
    "用简单易懂的语言解释相对论的核心思想，适合中学生阅读。",
    "请介绍大语言模型的预训练和微调过程，以及 RLHF 在其中的作用。",
    "写一个 Python 函数，给定一个链表，判断是否存在环，并返回环的起始节点。",
    "分析当前人工智能领域最前沿的研究方向，以及未来 5 年可能的突破点。",
]

SYSTEM_PROMPT = "你是一个专业、简洁的 AI 助手。请用中文回答问题，除非题目要求使用英文。"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def check_server(base_url: str, timeout: int = 5) -> bool:
    """返回 True 表示 200 OK；503 说明还在初始化，打印提示后返回 False。"""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        if resp.status_code == 200:
            return True
        if resp.status_code == 503:
            print(
                f"  服务器进程已启动但仍在初始化（/health 返回 503）。\n"
                f"  EAGLE 需要额外加载 draft 模型并 capture CUDA graph，\n"
                f"  请等待服务端日志出现 'fired up and ready' 后再重试。"
            )
        return False
    except requests.exceptions.ConnectionError:
        return False


def generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> dict:
    """调用 /generate 接口。"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=300)
    resp.raise_for_status()
    latency = time.perf_counter() - t0

    ret = resp.json()
    meta = ret.get("meta_info", {})
    tokens = meta.get("completion_tokens", 0)
    return {
        "text": ret.get("text", ""),
        "latency": latency,
        "tokens": tokens,
        "speed": tokens / latency if latency > 0 else 0.0,
        "meta": meta,
    }


def generate_chat(
    base_url: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    model: str = "default",
) -> dict:
    """调用 OpenAI-兼容 /v1/chat/completions 接口。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=300
    )
    resp.raise_for_status()
    latency = time.perf_counter() - t0

    ret = resp.json()
    text = ret["choices"][0]["message"]["content"]
    usage = ret.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    return {
        "text": text,
        "latency": latency,
        "tokens": tokens,
        "speed": tokens / latency if latency > 0 else 0.0,
    }


def warmup(base_url: str, chat: bool = False, rounds: int = 2):
    """预热 GPU（避免第一次推理的冷启动偏差）。"""
    print("  [预热] 发送预热请求...")
    for _ in range(rounds):
        if chat:
            generate_chat(base_url, "你好", max_tokens=10)
        else:
            generate(base_url, "你好", max_new_tokens=10)
    print("  [预热] 完成\n")


# ---------------------------------------------------------------------------
# 核心 Demo 逻辑
# ---------------------------------------------------------------------------

def run_single_request_demo(base_url: str, chat: bool = False):
    """演示单条请求的生成结果和速度。"""
    print("=" * 65)
    print("  单条请求演示")
    print("=" * 65)

    prompt = DEMO_PROMPTS[0]
    print(f"  Prompt: {prompt}\n")

    if chat:
        result = generate_chat(base_url, prompt)
    else:
        result = generate(base_url, prompt)

    print(f"  Generated:\n  {result['text']}\n")
    print(f"  Latency : {result['latency']:.3f} s")
    print(f"  Tokens  : {result['tokens']}")
    print(f"  Speed   : {result['speed']:.1f} token/s")
    print()


def run_batch_demo(base_url: str, chat: bool = False, num_prompts: int = 5):
    """顺序发送多条请求，统计平均速度。"""
    print("=" * 65)
    print(f"  批量请求演示（{num_prompts} 条 prompt，顺序发送）")
    print("=" * 65)

    prompts = DEMO_PROMPTS[:num_prompts]
    total_tokens, total_latency = 0, 0.0

    for i, prompt in enumerate(prompts, 1):
        if chat:
            r = generate_chat(base_url, prompt)
        else:
            r = generate(base_url, prompt)
        total_tokens += r["tokens"]
        total_latency += r["latency"]
        preview = r["text"][:60].replace("\n", " ")
        print(
            f"  [{i}/{num_prompts}] {prompt[:30]!r}…"
            f"\n           => {preview!r}"
            f"\n           ({r['latency']:.2f}s, {r['tokens']} tok, {r['speed']:.1f} tok/s)\n"
        )

    if total_latency > 0:
        print(
            f"  合计: {total_tokens} tokens, {total_latency:.2f} s, "
            f"平均 {total_tokens / total_latency:.1f} tok/s"
        )
    print()


def run_comparison(
    eagle_url: str,
    baseline_url: str,
    chat: bool = False,
    num_prompts: int = 5,
):
    """对比 EAGLE 和普通解码的速度。"""
    print("=" * 65)
    print("  EAGLE vs 基线 对比测试")
    print("=" * 65)

    prompts = DEMO_PROMPTS[:num_prompts]

    eagle_speeds, baseline_speeds = [], []

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] Prompt: {prompt[:40]!r}…")

        if chat:
            r_eagle = generate_chat(eagle_url, prompt)
            r_base = generate_chat(baseline_url, prompt)
        else:
            r_eagle = generate(eagle_url, prompt)
            r_base = generate(baseline_url, prompt)

        eagle_speeds.append(r_eagle["speed"])
        baseline_speeds.append(r_base["speed"])

        speedup = r_eagle["speed"] / r_base["speed"] if r_base["speed"] > 0 else 0
        print(
            f"    EAGLE   : {r_eagle['speed']:6.1f} tok/s  "
            f"({r_eagle['tokens']} tok, {r_eagle['latency']:.2f}s)"
        )
        print(
            f"    Baseline: {r_base['speed']:6.1f} tok/s  "
            f"({r_base['tokens']} tok, {r_base['latency']:.2f}s)"
        )
        print(f"    Speedup : {speedup:.2f}×\n")

    avg_eagle = sum(eagle_speeds) / len(eagle_speeds)
    avg_base = sum(baseline_speeds) / len(baseline_speeds)
    avg_speedup = avg_eagle / avg_base if avg_base > 0 else 0

    print("-" * 65)
    print(f"  平均 EAGLE   速度 : {avg_eagle:.1f} tok/s")
    print(f"  平均 Baseline 速度: {avg_base:.1f} tok/s")
    print(f"  平均加速比        : {avg_speedup:.2f}×")
    print(
        "\n  注：加速比受 batch size、序列长度、prompt 类型影响。"
        "\n  高重复性内容（如代码）通常接受率更高，加速比更大。"
    )
    print()


# ---------------------------------------------------------------------------
# 离线 Engine 模式
# ---------------------------------------------------------------------------

def run_offline_engine(
    target_model: str,
    draft_model: str,
    num_steps: int,
    topk: int,
    num_draft_tokens: int,
):
    """使用 sgl.Engine API 直接在进程内运行，无需启动服务器。"""
    print("=" * 65)
    print("  离线 Engine 模式（sgl.Engine API）")
    print("=" * 65)
    print(f"  Target model : {target_model}")
    print(f"  Draft model  : {draft_model}")
    print(
        f"  EAGLE params : steps={num_steps}, topk={topk}, "
        f"draft_tokens={num_draft_tokens}\n"
    )

    try:
        import sglang as sgl
    except ImportError:
        print("  ERROR: 请先安装 sglang: pip install sglang")
        return

    print("  正在加载模型（首次加载可能需要几分钟）...")
    llm = sgl.Engine(
        model_path=target_model,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=draft_model,
        speculative_num_steps=num_steps,
        speculative_eagle_topk=topk,
        speculative_num_draft_tokens=num_draft_tokens,
        cuda_graph_max_bs=8,
    )

    sampling_params = {"temperature": 0.0, "max_new_tokens": 200}
    prompts = DEMO_PROMPTS[:4]

    print(f"  发送 {len(prompts)} 条请求...\n")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.perf_counter() - t0

    for prompt, output in zip(prompts, outputs):
        text_preview = output["text"][:80].replace("\n", " ")
        print(f"  Prompt  : {prompt[:40]!r}…")
        print(f"  Output  : {text_preview!r}…")
        print()

    total_tokens = sum(
        o.get("meta_info", {}).get("completion_tokens", 0) for o in outputs
    )
    if total_time > 0 and total_tokens > 0:
        print(f"  总时间 : {total_time:.2f} s")
        print(f"  总 tokens : {total_tokens}")
        print(f"  吞吐量 : {total_tokens / total_time:.1f} tok/s")

    llm.shutdown()
    print()


# ---------------------------------------------------------------------------
# EAGLE 参数调优建议
# ---------------------------------------------------------------------------

def print_tuning_guide():
    print("=" * 65)
    print("  EAGLE 参数调优指南")
    print("=" * 65)
    print("""
  关键参数及其影响：

  speculative_num_steps（草稿步数，推荐 3-5）
    ├─ 步数越多 → 候选树越深 → 潜在可接受的 token 序列越长
    └─ 但步数过多 → 深层节点接受率↓ → 收益递减，还会增加内存压力

  speculative_eagle_topk（每步 top-k，推荐 4-8）
    ├─ topk 越大 → 候选宽度越大 → 接受率越高
    └─ 但 topk 越大 → KV cache 占用↑ → 对长序列或大 batch 不友好

  speculative_num_draft_tokens（最终候选数）
    └─ 通常设为 num_steps × topk 的子集（如 16），不必全选

  实践建议：
    • 对话/问答类（输出较确定）: steps=5, topk=4, draft_tokens=20
    • 代码生成（高重复性）     : steps=3, topk=4, draft_tokens=16
    • 创意写作（输出多样）     : steps=3, topk=8, draft_tokens=16
    • 内存紧张时              : 降低 topk，而非步数

  加速比不如预期时的排查思路：
    1. 确认 draft 模型与 target 匹配（相同架构系列）
    2. 查看 server 日志中的 spec_accept_rate（< 0.5 说明草稿质量差）
    3. 尝试增大 topk 或改用质量更好的 draft 模型
    4. 确保 cuda_graph_max_bs >= 实际 batch size
""")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_server_demo(args: argparse.Namespace):
    base_url = f"http://{args.host}:{args.port}"
    print("=" * 65)
    print("  SGLang EAGLE Speculative Decoding Demo（服务器模式）")
    print("=" * 65)
    print(f"  EAGLE server : {base_url}")
    print(f"  Chat API     : {args.chat}")
    if args.compare:
        baseline_url = f"http://{args.host}:{args.baseline_port}"
        print(f"  Baseline     : {baseline_url}")
    print()

    # 健康检查
    print("[1] 检查 EAGLE 服务器...")
    if not check_server(base_url):
        print(f"  ERROR: 无法连接到 {base_url}，请先启动服务器（见脚本顶部注释）")
        return
    print("  OK\n")

    if args.compare:
        baseline_url = f"http://{args.host}:{args.baseline_port}"
        print("[1b] 检查基线服务器...")
        if not check_server(baseline_url):
            print(f"  ERROR: 无法连接到基线服务器 {baseline_url}")
            return
        print("  OK\n")

    # 预热
    warmup(base_url, chat=args.chat)

    # 单条请求演示
    run_single_request_demo(base_url, chat=args.chat)

    # 批量请求演示
    run_batch_demo(base_url, chat=args.chat, num_prompts=args.num_prompts)

    # 对比测试
    if args.compare:
        warmup(baseline_url, chat=args.chat)
        run_comparison(base_url, baseline_url, chat=args.chat, num_prompts=args.num_prompts)

    print_tuning_guide()
    print("Demo 完成！")


def main():
    parser = argparse.ArgumentParser(
        description="SGLang EAGLE Speculative Decoding + Qwen Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000, help="EAGLE 服务器端口")
    parser.add_argument(
        "--baseline-port", type=int, default=30001, help="基线服务器端口（用于对比）"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="与基线对比（需同时启动两个服务器）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="使用 OpenAI /v1/chat/completions 接口（带 system prompt）",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4, help="测试用的 prompt 数量"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="使用离线 Engine API 模式（无需启动服务器）",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_TARGET_MODEL,
        help=f"Target 模型路径（离线模式用，默认: {DEFAULT_TARGET_MODEL}）",
    )
    parser.add_argument(
        "--draft-model-path",
        default=DEFAULT_DRAFT_MODEL,
        help=f"Draft 模型路径（离线模式用，默认: {DEFAULT_DRAFT_MODEL}）",
    )
    parser.add_argument(
        "--num-steps", type=int, default=EAGLE_NUM_STEPS, help="EAGLE 草稿步数"
    )
    parser.add_argument(
        "--topk", type=int, default=EAGLE_TOPK, help="EAGLE 每步 top-k"
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=EAGLE_NUM_DRAFT_TOKENS,
        help="EAGLE 最终候选 token 数",
    )

    args = parser.parse_args()

    if args.offline:
        # 必须在 __main__ 保护下运行，Engine 内部使用 spawn
        run_offline_engine(
            target_model=args.model_path,
            draft_model=args.draft_model_path,
            num_steps=args.num_steps,
            topk=args.topk,
            num_draft_tokens=args.num_draft_tokens,
        )
    else:
        run_server_demo(args)


# Engine 使用 spawn 创建子进程，必须有 __main__ 保护
if __name__ == "__main__":
    main()
