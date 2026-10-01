"""
STEP 5 -- Similarity baseline scorer. Make "what to keep" explicit and measurable.

Step 4's LLM decided what to keep inside a black box. To argue that a better rule
exists, you first need the standard rule written down as code you can measure.
The standard rule is similarity: score every candidate memory by cosine
similarity to the recent conversation, keep the top N.

This runs four policies on the same sessions so they can be compared directly:

  raw         keep everything (the upper bound on retention, lower bound on focus)
  recency     keep the newest N (if similarity can't beat this, it isn't helping)
  similarity  keep the N most similar to the last few turns   <-- THE BASELINE
  llm         Step 4's summariser (skipped in --dry-run)

What to look for:
  - which items similarity keeps after each session. Expect it to keep whatever
    the latest session was ABOUT, and drop quiet but important facts (the FYP
    fact, the drink, the study spot) once nobody is talking about them.
  - whether it keeps stored QUESTIONS. They look extremely similar to the topic
    being discussed, and contain no answer at all.
  - the control question (t19). Similarity has no reason to keep turn 1.

Every one of those is a fact being kept or dropped for looking related, not for
being useful. That is the argument your thesis makes, now as a number.

Run:  python step5_similarity_scorer.py --dry-run
      python step5_similarity_scorer.py
      python step5_similarity_scorer.py -n 3 --embedder local
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
    names = ["raw", "recency", "similarity"] + ([] if args.dry_run else ["llm"])
    results = run_policies(names, toy_events(), args, embedder, client)
    compare(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
