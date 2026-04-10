"""
SGLang dLLM (Diffusion LLM) 使用 Demo
======================================

本脚本演示如何启动 SGLang 服务并使用 dLLM 推理算法进行文本生成。

支持的模型架构:
  - SDARForCausalLM / SDARMoeForCausalLM  (block_size=4, mask_id=151669)
  - LLaDA2MoeModelLM                       (block_size=32, mask_id=156895)

支持的算法:
  - LowConfidence      : 基于置信度阈值的迭代去噪
  - JointThreshold     : 联合阈值策略
  - SmallDraftVerify   : 小模型起草 + 大模型验证（需额外指定 small_model_path）

使用方法
--------
Step 1: 启动服务器（在另一个终端中运行）

  # LowConfidence 算法（最简单，无需额外模型）
  python -m sglang.launch_server \\
      --model-path <your-model-path> \\
      --dllm-algorithm LowConfidence \\
      --port 30000

  # SmallDraftVerify 算法（需要指定小模型）
  python -m sglang.launch_server \\
      --model-path <your-large-model-path> \\
      --dllm-algorithm SmallDraftVerify \\
      --dllm-small-model-path <your-small-model-path> \\
      --port 30000

Step 2: 在本脚本所在目录运行本 demo

  python demo_dllm.py --host localhost --port 30000
"""

import argparse
import json
import time

import requests


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "请简要介绍一下量子计算的基本原理。",
    "写一段 Python 代码，实现快速排序算法。",
    "What is the difference between machine learning and deep learning?",
    "解释一下什么是 Transformer 架构，以及它在 NLP 中的应用。",
    "用简单的语言解释相对论的核心思想。",
    "写一个用于计算斐波那契数列的递归函数，并分析其时间复杂度。",
    "什么是大语言模型？它是如何被训练的？",
    "请介绍深度强化学习的基本原理和典型应用场景。",
]

DEFAULT_BATCH_SIZES = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def check_server(base_url: str, timeout: int = 5) -> bool:
    """检查服务器是否在线。"""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    stream: bool = False,
) -> dict:
    """
    调用 /generate 接口进行文本生成。

    返回包含 text、latency、speed 等信息的字典。
    """
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "stream": stream,
    }

    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        stream=stream,
        timeout=300,
    )
    response.raise_for_status()

    if stream:
        full_text = ""
        print("  [streaming] ", end="", flush=True)
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                delta = data["text"][len(full_text):]
                full_text = data["text"]
                print(delta, end="", flush=True)
        print()  # newline after stream
        latency = time.perf_counter() - t0
        return {"text": full_text, "latency": latency}

    ret = response.json()
    latency = time.perf_counter() - t0
    meta = ret.get("meta_info", {})
    tokens = meta.get("completion_tokens", 0)
    speed = tokens / latency if latency > 0 else 0.0
    return {
        "text": ret.get("text", ""),
        "latency": latency,
        "tokens": tokens,
        "speed": speed,
        "meta": meta,
    }


def generate_batch(
    base_url: str,
    prompts: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """
    调用 /generate 接口进行批量文本生成（单次请求发送多个 prompt）。

    返回包含每个 prompt 结果、总延迟、吞吐量等信息的字典。
    """
    payload = {
        "text": prompts,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    latency = time.perf_counter() - t0

    ret = response.json()
    # /generate 对批量请求返回列表
    if isinstance(ret, list):
        results = ret
    else:
        results = [ret]

    total_tokens = sum(r.get("meta_info", {}).get("completion_tokens", 0) for r in results)
    throughput = total_tokens / latency if latency > 0 else 0.0

    return {
        "results": results,
        "batch_size": len(prompts),
        "latency": latency,
        "total_tokens": total_tokens,
        "throughput": throughput,
    }


def run_batch_size_test(
    base_url: str,
    batch_sizes: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    repeats: int = 3,
) -> None:
    """
    针对不同 batch_size 进行吞吐量对比测试。

    每个 batch_size 重复 `repeats` 次，取平均值。
    """
    print("=" * 60)
    print("  Batch Size 对比测试")
    print("=" * 60)
    print(f"  max_new_tokens : {max_new_tokens}")
    print(f"  temperature    : {temperature}")
    print(f"  repeats        : {repeats}")
    print(f"  batch_sizes    : {batch_sizes}")
    print()

    summary = []

    for bs in batch_sizes:
        # 从 DEMO_PROMPTS 中循环取足够数量的 prompt
        prompts = [DEMO_PROMPTS[i % len(DEMO_PROMPTS)] for i in range(bs)]

        latencies = []
        throughputs = []

        print(f"  [batch_size={bs}] 开始测试...")
        for r in range(repeats):
            try:
                res = generate_batch(
                    base_url, prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                latencies.append(res["latency"])
                throughputs.append(res["throughput"])
                print(
                    f"    run {r+1}/{repeats}: latency={res['latency']:.3f}s  "
                    f"tokens={res['total_tokens']}  throughput={res['throughput']:.1f} tok/s"
                )
            except Exception as e:
                print(f"    run {r+1}/{repeats}: ERROR - {e}")

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            avg_thr = sum(throughputs) / len(throughputs)
            summary.append({
                "batch_size": bs,
                "avg_latency": avg_lat,
                "avg_throughput": avg_thr,
            })
            print(
                f"  [batch_size={bs}] 平均: latency={avg_lat:.3f}s  "
                f"throughput={avg_thr:.1f} tok/s\n"
            )
        else:
            print(f"  [batch_size={bs}] 所有请求均失败，跳过。\n")

    # 汇总表格
    if summary:
        print("-" * 60)
        print(f"  {'batch_size':>12}  {'avg_latency(s)':>16}  {'avg_throughput(tok/s)':>22}")
        print("-" * 60)
        for row in summary:
            print(
                f"  {row['batch_size']:>12}  {row['avg_latency']:>16.3f}  "
                f"{row['avg_throughput']:>22.1f}"
            )
        print("-" * 60)
        # 找出最高吞吐量的 batch_size
        best = max(summary, key=lambda x: x["avg_throughput"])
        print(
            f"\n  最高吞吐量: batch_size={best['batch_size']}  "
            f"({best['avg_throughput']:.1f} tok/s)"
        )
    print()


def generate_openai_compat(
    base_url: str,
    prompt: str,
    max_tokens: int = 128,
    model: str = "default",
) -> dict:
    """
    使用 OpenAI 兼容接口（/v1/completions）进行文本生成。

    需要 pip install openai
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先安装 openai: pip install openai")

    client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")
    t0 = time.perf_counter()
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    latency = time.perf_counter() - t0
    text = completion.choices[0].text
    tokens = completion.usage.completion_tokens if completion.usage else 0
    speed = tokens / latency if latency > 0 else 0.0
    return {"text": text, "latency": latency, "tokens": tokens, "speed": speed}


# ---------------------------------------------------------------------------
# Demo 主体
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("  SGLang dLLM Demo")
    print("=" * 60)
    print(f"  Server : {base_url}")
    print(f"  API    : {'OpenAI-compat' if args.openai else '/generate'}")
    print(f"  Stream : {args.stream}")
    print()

    # 1. 健康检查
    print("[1/3] 检查服务器状态 ...")
    if not check_server(base_url):
        print(
            f"  ERROR: 无法连接到服务器 {base_url}\n"
            "  请先按照脚本顶部注释中的说明启动 SGLang 服务器。"
        )
        return
    print("  OK - 服务器在线\n")

    # 2. 单条请求演示
    print("[2/3] 单条请求演示 ...")
    prompt = args.prompt or DEMO_PROMPTS[0]
    print(f"  Prompt: {prompt!r}\n")

    if args.openai:
        result = generate_openai_compat(
            base_url, prompt, max_tokens=args.max_new_tokens
        )
    else:
        result = generate(
            base_url,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )

    print(f"  Generated text:\n  {result['text']}\n")
    print(f"  Latency : {result['latency']:.3f} s")
    if "tokens" in result:
        print(f"  Tokens  : {result['tokens']}")
    if "speed" in result:
        print(f"  Speed   : {result['speed']:.1f} token/s")
    print()

    # 3. 批量请求演示（仅 /generate 接口）
    if not args.openai and not args.stream:
        print("[3/3] 批量请求演示（多个 prompt 依次发送）...")
        total_tokens = 0
        total_latency = 0.0
        for i, p in enumerate(DEMO_PROMPTS[:3], 1):
            r = generate(base_url, p, max_new_tokens=64, temperature=0.0)
            total_tokens += r.get("tokens", 0)
            total_latency += r["latency"]
            preview = r["text"][:80].replace("\n", " ")
            print(f"  [{i}] {p!r}")
            print(f"       => {preview!r}  ({r['latency']:.2f}s, {r.get('tokens',0)} tok)")
        print()
        if total_latency > 0:
            print(f"  总计: {total_tokens} tokens, {total_latency:.2f} s, "
                  f"平均 {total_tokens/total_latency:.1f} tok/s")
    else:
        print("[3/3] 跳过批量请求（流式/OpenAI 模式）")

    # 4. Batch size 对比测试
    if args.batch_size_test and not args.openai and not args.stream:
        print()
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        run_batch_size_test(
            base_url,
            batch_sizes=batch_sizes,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repeats=args.repeats,
        )

    print()
    print("Demo 完成！")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGLang dLLM 使用 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost", help="服务器地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=30000, help="服务器端口 (默认: 30000)")
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="最大生成 token 数 (默认: 128)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="采样温度 (默认: 0.0 = greedy)"
    )
    parser.add_argument(
        "--stream", action="store_true", help="启用流式输出（仅 /generate 接口）"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="使用 OpenAI 兼容接口 /v1/completions（需要 pip install openai）",
    )
    parser.add_argument(
        "--batch-size-test",
        action="store_true",
        help="启用不同 batch_size 的吞吐量对比测试（仅 /generate 接口）",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help=f"逗号分隔的 batch_size 列表（默认: {','.join(str(b) for b in DEFAULT_BATCH_SIZES)}）",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="每个 batch_size 的重复测试次数，用于取平均值（默认: 3）",
    )

    args = parser.parse_args()
    run_demo(args)
