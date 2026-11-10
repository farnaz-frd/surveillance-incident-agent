"""
Times three Ollama calls and splits each one into: model load, prompt
processing, and generation. Run from your venv:

    python diag_ollama_timing.py

Ollama reports its own timings (in nanoseconds) in every non-streaming
/api/generate response. This script only converts them to seconds.
"""

import os
import time

import requests

BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = "qwen2.5:3b"
NS = 1_000_000_000  # nanoseconds per second


def timed_call(label: str, prompt: str) -> None:
    start = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            # Cap the output so the test mostly measures load + prompt reading.
            "options": {"num_predict": 20},
        },
        timeout=900,
    )
    wall_s = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()

    load_s = data.get("load_duration", 0) / NS
    prompt_tokens = data.get("prompt_eval_count", 0)
    prompt_s = data.get("prompt_eval_duration", 0) / NS
    gen_tokens = data.get("eval_count", 0)
    gen_s = data.get("eval_duration", 0) / NS

    print(f"--- {label} ---")
    print(f"  total wall clock:   {wall_s:7.1f} s")
    print(f"  model load:         {load_s:7.1f} s")
    print(f"  prompt processing:  {prompt_s:7.1f} s   ({prompt_tokens} tokens)")
    if prompt_s > 0:
        print(f"                      {prompt_tokens / prompt_s:7.1f} tokens/s")
    print(f"  generation:         {gen_s:7.1f} s   ({gen_tokens} tokens)")
    if gen_s > 0:
        print(f"                      {gen_tokens / gen_s:7.1f} tokens/s")
    print()


if __name__ == "__main__":
    long_prompt = " ".join(
        f"Frame {i}: a person stands near a glass door next to a badge reader."
        for i in range(150)
    )
    long_prompt += "\nIn one short sentence, what is this text about?"

    timed_call("Call 1: tiny prompt (model may need loading)", "hi")
    timed_call("Call 2: tiny prompt again (model should be warm now)", "hello there")
    timed_call("Call 3: long prompt (a few thousand tokens)", long_prompt)