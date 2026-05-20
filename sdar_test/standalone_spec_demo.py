"""
SGLang Standalone Speculative Decoding Demo
=============================================

Standalone Speculative Decoding 与 EAGLE 的核心区别
-----------------------------------------------------
  EAGLE    : draft model 经过专门训练，接收 target 的 hidden state 作为输入，
             预测质量高（接受率 70-90%），但需要配套的专用 draft 模型。

  Standalone: draft model 是普通语言模型（无需特殊训练），自回归生成候选 token，
              target model 逐 token 验证并做 rejection sampling。
              接受率较低（40-70%），但优势是：
                - 任何小模型都可以作为 draft（同系列更佳）
                - 不依赖专用训练权重，模型选择灵活

工作流程（每个 decode 轮次）
-----------------------------
  1. Draft model 自回归生成 K 个候选 token（speculative_num_steps 步）
  2. Target model 一次前向，对这 K 个 token 做并行验证
  3. Rejection sampling：从前往后，接受概率 p_target(t) / p_draft(t) 的 token
  4. 第一个被拒绝的位置重采样一个 bonus token，之后丢弃剩余 draft token
  5. 最终接受 K' 个 token（1 ≤ K' ≤ K+1），相当于 1 次 target forward 产出多 token

推荐模型对（均可在单张 H100 80GB 上运行）
------------------------------------------
  性价比最高（同系列，词表相同，接受率最高）：
    Target : Qwen/Qwen2.5-7B-Instruct
    Draft  : Qwen/Qwen2.5-0.5B-Instruct  （0.5B，极小开销）
    Draft  : Qwen/Qwen2.5-1.5B-Instruct  （1.5B，接受率稍高）

  LLaMA 系列：
    Target : meta-llama/Meta-Llama-3.1-8B-Instruct
    Draft  : meta-llama/Meta-Llama-3.2-1B-Instruct

  注：同系列（相同分词器、相同训练数据分布）接受率比跨系列高得多。

关键参数说明
------------
  --speculative-algorithm STANDALONE     固定，区别于 EAGLE
  --speculative-draft-model-path         普通小模型路径（无需特殊训练）
  --speculative-num-steps                每轮 draft 生成的候选 token 数（推荐 4-8）
  --speculative-eagle-topk               设为 1（standalone 只用 top-1 draft token）
  --speculative-num-draft-tokens         同 num-steps（standalone 线性生成）

使用方法
--------
Step 1: 启动 Standalone 推测解码服务器

  python -m sglang.launch_server \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --speculative-algorithm STANDALONE \\
      --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \\
      --speculative-num-steps 5 \\
      --speculative-eagle-topk 1 \\
      --speculative-num-draft-tokens 5 \\
      --cuda-graph-max-bs 8 \\
      --port 30000

Step 2: 启动基线服务器（同 target 模型，无推测解码）

  python -m sglang.launch_server \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --port 30001

Step 3: 运行 demo

  # 只测 standalone spec decoding
  python standalone_spec_demo.py --port 30000

  # 与基线对比（推荐，能看到实际加速比）
  python standalone_spec_demo.py --port 30000 --baseline-port 30001 --compare

  # 测试不同 draft 步数对接受率的影响
  python standalone_spec_demo.py --port 30000 --show-server-tip
"""

import argparse
import time

import requests

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SPEC_NUM_STEPS = 5   # 每轮生成的 draft token 数

# 覆盖不同场景的测试 prompt（尽量短输入、长输出，让 decode 阶段占主导）
DEMO_PROMPTS = [
    "请详细解释快速排序算法的工作原理，并用 Python 实现一个完整的示例代码。",
    "Write a Python function that implements binary search on a sorted list. Include docstring and test cases.",
    "解释大语言模型中注意力机制的数学原理，包括 Query、Key、Value 的计算过程。",
    "Describe the differences between TCP and UDP protocols, with examples of when to use each.",
    "用 Python 实现一个简单的 HTTP 服务器，支持 GET 和 POST 请求。",
    "What are the SOLID principles in software engineering? Give a concrete example for each.",
    "解释 Docker 容器和虚拟机的主要区别，以及各自的适用场景。",
    "Implement a thread-safe singleton pattern in Python with explanation.",
]

SYSTEM_PROMPT = "You are a helpful assistant. Be concise and accurate."


# ---------------------------------------------------------------------------
# HTTP 工具函数
# ---------------------------------------------------------------------------

def check_server(base_url: str, timeout: int = 5) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        if resp.status_code == 200:
            return True
        if resp.status_code == 503:
            print(
                "  服务器进程已启动但仍在初始化（/health 返回 503）。\n"
                "  请等待服务端日志出现 'fired up and ready' 后再重试。"
            )
        return False
    except requests.exceptions.ConnectionError:
        return False


def generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": temperature},
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


def warmup(base_url: str, rounds: int = 3):
    print("  [预热] 发送预热请求...", end="", flush=True)
    for _ in range(rounds):
        generate(base_url, "Hello", max_new_tokens=16)
    print(" 完成\n")


# ---------------------------------------------------------------------------
# Demo 核心逻辑
# ---------------------------------------------------------------------------

def run_single_demo(base_url: str, max_tokens: int = 256):
    print("=" * 65)
    print("  单条请求演示")
    print("=" * 65)
    prompt = DEMO_PROMPTS[0]
    print(f"  Prompt: {prompt}\n")
    r = generate(base_url, prompt, max_new_tokens=max_tokens)
    print(f"  Generated:\n{r['text']}\n")
    print(f"  Latency : {r['latency']:.3f} s")
    print(f"  Tokens  : {r['tokens']}")
    print(f"  Speed   : {r['speed']:.1f} tok/s")
    print()


def run_batch_demo(base_url: str, num_prompts: int = 6, max_tokens: int = 256):
    print("=" * 65)
    print(f"  批量请求演示（{num_prompts} 条，顺序发送）")
    print("=" * 65)
    prompts = DEMO_PROMPTS[:num_prompts]
    total_tok, total_lat = 0, 0.0
    for i, p in enumerate(prompts, 1):
        r = generate(base_url, p, max_new_tokens=max_tokens)
        total_tok += r["tokens"]
        total_lat += r["latency"]
        preview = r["text"][:55].replace("\n", " ")
        print(
            f"  [{i}/{num_prompts}] {p[:35]!r}…\n"
            f"           => {preview!r}…\n"
            f"           ({r['latency']:.2f}s, {r['tokens']} tok, {r['speed']:.1f} tok/s)\n"
        )
    if total_lat > 0:
        print(f"  合计: {total_tok} tokens, {total_lat:.2f} s, 平均 {total_tok/total_lat:.1f} tok/s")
    print()


def run_comparison(
    spec_url: str,
    baseline_url: str,
    num_prompts: int = 6,
    max_tokens: int = 256,
):
    print("=" * 65)
    print("  Standalone Spec Decoding vs 基线 对比")
    print("=" * 65)
    prompts = DEMO_PROMPTS[:num_prompts]
    spec_speeds, base_speeds = [], []

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] {prompt[:45]!r}…")
        r_spec = generate(spec_url, prompt, max_new_tokens=max_tokens)
        r_base = generate(baseline_url, prompt, max_new_tokens=max_tokens)
        spec_speeds.append(r_spec["speed"])
        base_speeds.append(r_base["speed"])
        speedup = r_spec["speed"] / r_base["speed"] if r_base["speed"] > 0 else 0
        print(
            f"    Standalone: {r_spec['speed']:6.1f} tok/s  "
            f"({r_spec['tokens']} tok, {r_spec['latency']:.2f}s)"
        )
        print(
            f"    Baseline  : {r_base['speed']:6.1f} tok/s  "
            f"({r_base['tokens']} tok, {r_base['latency']:.2f}s)"
        )
        print(f"    Speedup   : {speedup:.2f}×\n")

    avg_spec = sum(spec_speeds) / len(spec_speeds)
    avg_base = sum(base_speeds) / len(base_speeds)
    avg_speedup = avg_spec / avg_base if avg_base > 0 else 0
    print("-" * 65)
    print(f"  平均 Standalone 速度 : {avg_spec:.1f} tok/s")
    print(f"  平均 Baseline   速度 : {avg_base:.1f} tok/s")
    print(f"  平均加速比           : {avg_speedup:.2f}×")
    print()
    _print_interpretation(avg_speedup)


def _print_interpretation(speedup: float):
    """根据实际加速比给出解读和建议。"""
    print("  结果解读：")
    if speedup >= 1.5:
        print(f"  ✅ 加速效果显著（{speedup:.2f}×），standalone spec decoding 在此场景下工作良好。")
    elif speedup >= 1.0:
        print(
            f"  ⚠️  有一定加速（{speedup:.2f}×），但不够理想。\n"
            "     建议：增大 --speculative-num-steps（试试 8），或换更接近 target 的 draft 模型。"
        )
    else:
        print(
            f"  ❌ 加速比 < 1（{speedup:.2f}×），spec decoding 开销超过收益。\n"
            "     常见原因：\n"
            "       1. draft 模型接受率太低（查看服务端 accept rate）\n"
            "       2. 目标模型太小（baseline 本身已经很快）\n"
            "       3. draft 模型与 target 系列不匹配（词表不同）\n"
            "     建议：换更大的 target 模型（14B+），或换同系列 draft 模型。"
        )
    print()


def print_server_tip():
    """打印关键的服务端监控指标说明。"""
    print("=" * 65)
    print("  如何判断 Standalone Spec Decoding 是否正常工作")
    print("=" * 65)
    print("""
  查看 EAGLE/Standalone 服务端日志中的 Decode batch 行：

    Decode batch, ..., accept len: X.XX, accept rate: X.XX, ...

  关键指标：
    accept len  : 每轮接受的 token 总数（含 bonus token）
                  正常工作：2.0 ~ 5.0+
                  完全失效：1.0（仅 bonus token，draft 全被拒绝）

    accept rate : draft token 的接受率（0~1）
                  同系列小模型 draft：0.40 ~ 0.70
                  EAGLE 专用 draft ：0.70 ~ 0.90
                  完全失效         ：0.00

  Standalone vs EAGLE accept rate 对比：
    Standalone (Qwen2.5-0.5B → 7B) : 预期 0.40-0.60
    Standalone (Qwen2.5-1.5B → 7B) : 预期 0.50-0.70
    EAGLE (专用 draft → 7B)         : 预期 0.70-0.90

  加速比估算：
    accept_len = 3.0  →  理论最大加速 ≈ 3× （实际约 1.5-2× 因有其他开销）
    accept_len = 1.5  →  理论最大加速 ≈ 1.5×（实际约 1.0-1.2×）
    accept_len = 1.0  →  无加速，反而更慢
""")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SGLang Standalone Speculative Decoding Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000, help="Standalone spec 服务器端口")
    parser.add_argument("--baseline-port", type=int, default=30001, help="基线服务器端口")
    parser.add_argument("--compare", action="store_true", help="与基线对比")
    parser.add_argument("--num-prompts", type=int, default=6, help="测试 prompt 数量")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument(
        "--show-server-tip", action="store_true",
        help="打印服务端监控指标说明（accept len / accept rate）"
    )
    args = parser.parse_args()

    spec_url = f"http://{args.host}:{args.port}"

    print("=" * 65)
    print("  SGLang Standalone Speculative Decoding Demo")
    print("=" * 65)
    print(f"  Spec server : {spec_url}")
    if args.compare:
        print(f"  Baseline    : http://{args.host}:{args.baseline_port}")
    print()

    # 健康检查
    print("[1] 检查 Spec 服务器...")
    if not check_server(spec_url):
        print(f"  ERROR: 无法连接到 {spec_url}")
        print("  请先按照脚本顶部注释启动服务器。")
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
    warmup(spec_url)

    # 单条演示
    run_single_demo(spec_url, max_tokens=args.max_tokens)

    # 批量演示
    run_batch_demo(spec_url, num_prompts=args.num_prompts, max_tokens=args.max_tokens)

    # 对比测试
    if args.compare:
        warmup(baseline_url)
        run_comparison(
            spec_url, baseline_url,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
        )

    # 服务端指标说明
    if args.show_server_tip or args.compare:
        print_server_tip()

    print("Demo 完成！")
    print(
        "\n提示：运行时查看 EAGLE/Standalone 服务端日志中的 'accept len' 和 'accept rate'，"
        "\n      这是判断 spec decoding 是否真正工作的最直接指标。"
    )


if __name__ == "__main__":
    main()
