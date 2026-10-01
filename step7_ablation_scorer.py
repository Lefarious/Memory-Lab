"""
STEP 7 (stretch) -- Reasoning-aligned scorer stub. The first draft of your contribution.

Steps 4 and 5 keep memories for looking important (LLM judgement) or looking
related (similarity). This step keeps memories for being USEFUL, measured directly:

  1. Generate a few probe questions the candidate memories should answer.
  2. Remove each candidate in turn and re-ask every probe.
  3. utility = how many probes go from right to wrong when that item is removed.
  4. Keep the N items with the highest utility.

That is an ablation: you learn what a piece contributes by taking it away.

It is a STUB, deliberately. Things to notice and write about:
  - COST. It makes (candidates x probes) answer calls per session -- the run
    prints the total. Compare with similarity's zero. This is the speed vs
    accuracy trade-off from your literature review, appearing in your own code.
  - DUPLICATES. Two copies of the same fact each get utility 0, because each
    covers for the other. Leave-one-out cannot see redundancy. How would you fix
    it? (Remove clusters, not items? Score sequentially?)
  - PROBE SOURCE. The probes come from the same LLM that is being scored. If the
    probes miss a topic (say, the FYP fact), that topic gets utility 0 and is
    dropped. Where should good probes come from in a real system?
  - TIES. With few probes, most items tie at 0 and the tie-break (recency)
    quietly decides. Check how many items actually scored above 0.

Run:  python step7_ablation_scorer.py --dry-run   # ablation falls back to similarity
      python step7_ablation_scorer.py
      python step7_ablation_scorer.py -n 3
"""

from __future__ import annotations

import argparse
import sys

from consolidation import add_common_args, compare, run_policies, setup, toy_events


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--probes", type=int, default=4, help="probe questions per session")
    args = ap.parse_args()
    from consolidation import AblationPolicy
    AblationPolicy.probes_per_session = args.probes
    embedder, client = setup(args)
    results = run_policies(["similarity", "ablation"], toy_events(), args, embedder, client)
    compare(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
