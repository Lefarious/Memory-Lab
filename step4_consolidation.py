"""
STEP 4 -- Consolidation. Between sessions, compress raw turns into a few durable facts.

Step 3 showed the store growing forever and retrieval slots filling up with stale
facts, decoys and old questions. The obvious fix is the one most production systems
use: at the end of each session, have an LLM rewrite everything into at most N
short facts and throw the raw turns away.

This script runs the same Step 3 conversation, now split into four sessions
(turns 1-6, 7-10, 11-14, 15-19), under two policies:

  raw   the Step 3 behaviour -- nothing is ever removed
  llm   after each session, an LLM keeps <= N facts (default N=5)

What to look for:
  - the facts the LLM keeps after each session (printed). Did it drop the stale
    editor? Did it keep the FYP fact, which nobody has mentioned since turn 1?
  - questions answered from the consolidated store vs the raw store
  - with -n 3: the LLM must now drop real facts. Which ones does it choose, and why?
    It has no way to know which facts future questions will need. It decides by
    rules of thumb -- that gap is exactly what your Step 7 scorer targets.

DECISION POINT: after this step, settle the framing. Steps 0-4 work under either
framing; Steps 5-7 below assume the reasoning-aligned consolidation framing.

Run:  python step4_consolidation.py --dry-run   # llm policy falls back to recency
      python step4_consolidation.py
      python step4_consolidation.py -n 3
"""

from __future__ import annotations

import argparse
import sys

from consolidation import add_common_args, compare, run_policies, setup, toy_events


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    embedder, client = setup(args)
    results = run_policies(["raw", "llm"], toy_events(), args, embedder, client)
    compare(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
