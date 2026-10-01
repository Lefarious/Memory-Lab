"""
STEP 6 -- Evaluate on a LoCoMo slice. Your first real number.

Everything so far ran on a 19-turn conversation you wrote yourself, which means
you also chose its traps. LoCoMo is a public benchmark of long two-person
conversations (~19-35 sessions, ~300-700 turns each) with question-answer pairs
written by other people. Same engine, same policies, someone else's test.

How the slice works:
  - pick one conversation (--conv) and the first few sessions (--sessions)
  - every turn is stored as "<speaker>: <text>", dated with its session date
  - the chosen policy consolidates at the end of each session
  - AFTER all sessions, ask the QA pairs whose evidence lies inside the slice
  - grade with token F1 (LoCoMo's standard), or --judge for LLM-as-judge

Question categories are numbered 1-5 in the data. The LoCoMo evaluation code
treats 5 as adversarial (no answer in the conversation), and these are skipped by
default. Check the LoCoMo paper/repository for what categories 1-4 mean before
you name them in your write-up.

What to look for:
  - raw vs similarity vs llm on the same questions
  - which categories collapse under consolidation. Questions about WHEN things
    happened are a good guess: summaries tend to drop dates.
  - why LoCoMo alone isn't enough: it tests recalling what was said, not using
    memory to act across sessions. That gap is what MemoryArena measures.

Cost: answering ~40 questions per policy with Haiku is a few cents. The llm
policy adds one call per session. Avoid the ablation policy here at first: it
makes hundreds of calls per session at this scale.

Run:  python step6_locomo_eval.py --dry-run                 # builds the slice, no model calls
      python step6_locomo_eval.py                           # raw + similarity + llm
      python step6_locomo_eval.py --sessions 8 -n 40 --judge
      python step6_locomo_eval.py --policies raw,recency,similarity --embedder local
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from consolidation import (
    Event, Item, Question, add_common_args, call, compare, run_policies, setup,
)
from memories import _m

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


# ---- data ----------------------------------------------------------------

def load_locomo(path: Path) -> list[dict]:
    if not path.exists():
        print(f"downloading LoCoMo to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_date(s: str) -> tuple[int, int, int]:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%d %B, %Y"):
        try:
            d = datetime.strptime(s.strip(), fmt)
            return d.year, d.month, d.day
        except ValueError:
            continue
    return 2023, 1, 1


def evidence_sessions(evidence: list[str]) -> set[int]:
    out = set()
    for e in evidence:
        out.update(int(x) for x in re.findall(r"D(\d+):", str(e)))
    return out


# ---- grading -------------------------------------------------------------

def normalise(text: str) -> list[str]:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return text.split()


def f1(prediction: str, gold: str) -> float:
    p, g = normalise(prediction), normalise(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: CORRECT or WRONG."


def make_judge(client, question: str, gold: str):
    def grade(answer: str) -> float:
        prompt = (f"Question: {question}\nGold answer: {gold}\nModel answer: {answer}\n\n"
                  "Is the model answer correct? It does not need to match word for word, "
                  "but it must contain the same key information (same date, same entity).")
        return 1.0 if call(client, JUDGE_SYSTEM, prompt, max_tokens=5).upper().startswith("CORRECT") else 0.0
    return grade


# ---- slice -> events ------------------------------------------------------

def build_events(sample: dict, max_sessions: int, categories: set[int], max_q: int,
                 judge_client=None) -> tuple[list[Event], int, int]:
    conv = sample["conversation"]
    n_sessions = 0
    n_turns = 0
    events: list[Event] = []
    s = 1
    while f"session_{s}" in conv and s <= max_sessions:
        date = parse_date(conv.get(f"session_{s}_date_time", ""))
        for i, turn in enumerate(conv[f"session_{s}"]):
            text = f"{turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f" [shares an image: {turn['blip_caption']}]"
            events.append(Event("turn", item=Item(_m(s * 1000 + i, *date, text), date)))
            n_turns += 1
        events.append(Event("end_session", date=date))
        n_sessions = s
        s += 1

    asked = 0
    for qa in sample["qa"]:
        if asked >= max_q:
            break
        cat = qa.get("category")
        if cat not in categories or "answer" not in qa:
            continue
        ev = evidence_sessions(qa.get("evidence", []))
        if not ev or max(ev) > n_sessions:
            continue                       # evidence lies outside the slice
        gold = str(qa["answer"])
        grade = (make_judge(judge_client, qa["question"], gold) if judge_client
                 else (lambda a, gold=gold: f1(a, gold)))
        events.append(Event("question", question=Question(
            text=qa["question"], grade=grade, label=f"c{cat}", gold=gold)))
        asked += 1
    return events, n_turns, asked


def by_category(results: dict[str, list[dict]]) -> None:
    print("\nmean score by category")
    names = list(results)
    print(f"{'category':<10}" + "".join(f"{n:>12}" for n in names) + f"{'count':>8}")
    cats = sorted({r["label"] for r in results[names[0]]})
    for c in cats:
        cells = []
        for n in names:
            sc = [r["score"] for r in results[n] if r["label"] == c and r["score"] is not None]
            cells.append(f"{(sum(sc) / len(sc)) if sc else float('nan'):>12.2f}")
        count = sum(1 for r in results[names[0]] if r["label"] == c)
        print(f"{c:<10}" + "".join(cells) + f"{count:>8}")


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.set_defaults(k=10, max_facts=30)
    ap.add_argument("--data", default="locomo10.json", help="path to locomo10.json")
    ap.add_argument("--conv", type=int, default=0, help="which of the 10 conversations")
    ap.add_argument("--sessions", type=int, default=5, help="how many sessions to feed in")
    ap.add_argument("--max-questions", type=int, default=40)
    ap.add_argument("--categories", default="1,2,3,4", help="comma list; 5 = adversarial")
    ap.add_argument("--policies", default=None, help="comma list, e.g. raw,similarity,llm")
    ap.add_argument("--judge", action="store_true", help="grade with LLM-as-judge instead of F1")
    ap.add_argument("--show-answers", action="store_true", help="print gold vs predicted")
    args = ap.parse_args()

    embedder, client = setup(args)
    sample = load_locomo(Path(args.data))[args.conv]
    a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
    cats = {int(c) for c in args.categories.split(",")}
    events, n_turns, n_q = build_events(sample, args.sessions, cats, args.max_questions,
                                        client if (args.judge and client) else None)

    print(f"conversation {args.conv}: {a} & {b} | sessions 1-{args.sessions} | "
          f"{n_turns} turns | {n_q} questions | k={args.k} | N={args.max_facts} | "
          f"grading: {'LLM judge' if args.judge and client else 'token F1'}")

    names = (args.policies.split(",") if args.policies
             else ["raw", "similarity"] + ([] if args.dry_run else ["llm"]))
    system = (f"You answer questions about a conversation between {a} and {b}, using only "
              "the memories provided. Each memory is dated; use the dates for questions "
              "about when things happened. If the memories do not contain the answer, say "
              "exactly: I don't know. Answer with a short phrase, not a sentence.")
    results = run_policies(names, events, args, embedder, client, system=system,
                           subject=f"{a} and {b}")

    if args.dry_run:
        print("\nDry run: slice built and every policy executed; no answers to grade.")
        return 0
    compare(results)
    by_category(results)
    if args.show_answers:
        for n in names:
            print(f"\n--- {n} ---")
            for r in results[n]:
                print(f"[{r['label']} {r['score']:.2f}] {r['q']}\n   gold: {r['gold']}\n   pred: {r['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
