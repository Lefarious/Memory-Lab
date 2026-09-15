"""
STEP 0 -- Prove the environment works before building anything on top of it.

Two checks:
  1. The numeric libraries Step 1 needs are importable.
  2. One round trip to an LLM completes and returns text.

Run:  export ANTHROPIC_API_KEY=sk-ant-...
      python step0_smoke_test.py

If check 2 fails, everything from Step 2 onward will fail in confusing ways.
Fix it here, where the error message is one line long.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import sys

MODEL = "claude-sonnet-4-6"


def check_imports() -> bool:
    ok = True
    for mod in ("numpy", "sklearn"):
        try:
            __import__(mod)
            print(f"  ok    {mod}")
        except ImportError:
            print(f"  MISSING  {mod}   -> pip install -r requirements.txt")
            ok = False
    try:
        __import__("anthropic")
        print("  ok    anthropic")
    except ImportError:
        print("  MISSING  anthropic   -> pip install anthropic")
        ok = False
    return ok


def check_llm_call() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  MISSING  ANTHROPIC_API_KEY not set in the environment.")
        return False

    import anthropic

    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly the word: pong",
                }
            ],
        )
    except Exception as exc:
        print(f"  FAILED   {type(exc).__name__}: {exc}")
        return False

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    print(f"  ok    model replied: {text!r}")
    print(f"  ok    tokens in/out: {resp.usage.input_tokens}/{resp.usage.output_tokens}")
    return True


def main() -> int:
    print("Step 0 -- environment check")
    print("-" * 50)
    print("imports:")
    imports_ok = check_imports()
    print("llm call:")
    llm_ok = check_llm_call()
    print("-" * 50)

    if imports_ok and llm_ok:
        print("PASS -- go to step1_retrieval.py")
        return 0
    print("FAIL -- fix the lines marked MISSING or FAILED above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
