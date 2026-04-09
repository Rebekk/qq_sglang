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
]


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
        for i, p in enumerate(DEMO_PROMPTS, 1):
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

    args = parser.parse_args()
    run_demo(args)
