"""
STEP 3 -- Multi-turn agent loop with a memory store that only ever grows.

Steps 1-2 used a fixed, hand-written store. Real agents don't have one. They build
their store as they go, one turn at a time. This step is the naive version of that:

    for each user turn:
        1. RETRIEVE  top-k memories relevant to this turn (before storing it)
        2. ANSWER    if the turn is a question, answer from those memories only
        3. STORE     append the raw turn to the store -- always, no exceptions

Nothing is ever updated, merged or deleted. Statements, restatements, corrections,
throwaway remarks and even the questions themselves all pile up. That is the
deliberate contrast with Mem0, which compares each new fact against similar stored
memories and decides ADD / UPDATE / DELETE / NOOP. Here the decision is always ADD.

The scripted conversation contains, on purpose:
  - SUPERSEDED facts   an old fact replaced by a newer one (editor, study spot,
                       embedder, drink)
  - DECOYS             turns that mention the old value without stating it as
                       current ("VS Code still has the better debugger")
  - RESTATEMENTS       the same fact said again in different words
  - REPEAT QUESTIONS   a question asked twice -- the first copy gets stored and
                       becomes a near-perfect match for the second
  - A CONTROL          one fact that never changes, so you can tell growth
                       damage apart from a broken pipeline

What to watch: as the store grows, how many of the k retrieval slots are taken by
stale facts, decoys and stored questions instead of the current fact. That crowding
is the problem Step 4 (consolidation) exists to fix.

Run:  python step3_agent_loop.py --dry-run         # retrieval only, no API calls
      python step3_agent_loop.py                   # answers question turns with the model
      python step3_agent_loop.py -k 1              # tighter retrieval budget
      python step3_agent_loop.py --embedder tfidf  # force a specific embedder
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from embedder import get_embedder
from memories import Memory, _m
from step1_retrieval import MemoryStore

MODEL = os.environ.get("MEMORY_LAB_MODEL", "claude-haiku-4-5-20251001")

SYSTEM = (
    "You are Joel's assistant. Answer using only the memories provided. "
    "Each memory is timestamped. If the memories do not contain the answer, "
    "say exactly: I don't know. Do not guess. Keep the answer to one short sentence."
)

ID_OFFSET = 100  # turn n is stored as memory id 100+n, clear of Step 1's ids


@dataclass
class Turn:
    n: int
    text: str
    # Question-only fields:
    expect: str | None = None  # keyword a correct answer must contain
    correct: int | None = None  # turn number holding the CURRENT fact
    stale: list[int] = field(default_factory=list)  # superseded turns + decoys

    @property
    def is_question(self) -> bool:
        return self.expect is not None


# fmt: off
CONVERSATION = [
    Turn(1,  "Quick context: my FYP is about memory systems for long-running LLM agents."),
    Turn(2,  "I write all my code in VS Code."),
    Turn(3,  "I study in the university library most days."),
    Turn(4,  "My Step 1 baseline used TF-IDF embeddings."),
    Turn(5,  "Which editor do I use?", expect="vs code", correct=2),
    Turn(6,  "I've been drinking a lot of coffee during long coding sessions."),
    Turn(7,  "I switched from VS Code to Neovim this week."),
    Turn(8,  "Honestly VS Code still has the better debugger though."),
    Turn(9,  "I moved my study spot from the library to home because of the commute."),
    Turn(10, "I switched my embedder from TF-IDF to sentence-transformers."),
    Turn(11, "Which editor do I use now?", expect="neovim", correct=7, stale=[2, 5, 8]),
    Turn(12, "I've cut out coffee and drink tea now."),
    Turn(13, "The library wifi was always slow anyway."),
    Turn(14, "Still on Neovim, the keybindings are growing on me."),
    Turn(15, "Where do I study?", expect="home", correct=9, stale=[3, 13]),
    Turn(16, "Which embedder am I using?", expect="sentence-transformers", correct=10, stale=[4]),
    Turn(17, "What do I drink while coding?", expect="tea", correct=12, stale=[6]),
    Turn(18, "Which editor do I use?", expect="neovim", correct=7, stale=[2, 5, 8, 11]),
    Turn(19, "What is my FYP about?", expect="memory", correct=1),  # control
]
# fmt: on


def to_memory(t: Turn) -> Memory:
    # One day per turn, so timestamps carry real ordering information.
    return _m(ID_OFFSET + t.n, 2026, 9, t.n, t.text)


def build_prompt(question: str, memories: list[Memory]) -> str:
    block = "\n".join(f"- [{m.when}] {m.text}" for m in memories) or "(no memories available)"
    return f"Memories:\n{block}\n\nQuestion: {question}"


def ask(client, question: str, memories: list[Memory]) -> tuple[str, int]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, memories)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.usage.input_tokens


def classify(mem_id: int, t: Turn, by_turn: dict[int, Turn]) -> str:
    n = mem_id - ID_OFFSET
    if n == t.correct:
        return "CURRENT"
    if n in t.stale:
        return "stale"
    if by_turn[n].is_question:
        return "question"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="retrieval only, no API calls")
    ap.add_argument("-k", type=int, default=3, help="memories retrieved per turn")
    ap.add_argument("--embedder", default=None, help="force an embedder (e.g. tfidf, local)")
    args = ap.parse_args()

    embedder = get_embedder(args.embedder) if args.embedder else get_embedder()
    by_turn = {t.n: t for t in CONVERSATION}

    client = None
    if not args.dry_run:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ANTHROPIC_API_KEY not set. Use --dry-run to see retrieval for free.")
            return 1
        import anthropic

        client = anthropic.Anthropic(api_key=key)

    stored: list[Memory] = []
    results = []  # one row per question turn
    store_name = "?"

    print(f"model    : {MODEL if not args.dry_run else '(dry run)'}")
    print(f"k        : {args.k}")
    print("=" * 78)

    for t in CONVERSATION:
        tag = "Q" if t.is_question else "S"
        print(f"\n[turn {t.n:>2} | store={len(stored):>2}] {tag}: {t.text}")

        if t.is_question and stored:
            # Deliberately unoptimised: rebuild the store each turn so TF-IDF
            # refits on the current vocabulary. Fine at this scale.
            store = MemoryStore(embedder)
            store.add_all(stored)
            store_name = store.embedder.name

            ranking = [m for _, m in store.search(t.text, k=len(stored))]
            scores = {m.id: s for s, m in store.search(t.text, k=len(stored))}
            top = ranking[: args.k]
            rank = next(
                (i + 1 for i, m in enumerate(ranking) if m.id == ID_OFFSET + t.correct), None
            )

            for i, m in enumerate(top, 1):
                label = classify(m.id, t, by_turn)
                print(f"     {i}. {scores[m.id]:.3f} [{label:>8}] t{m.id - ID_OFFSET}: {m.text}")

            labels = [classify(m.id, t, by_turn) for m in top]
            row = {
                "turn": t.n,
                "size": len(stored),
                "rank": rank,
                "crowd": sum(1 for x in labels if x in ("stale", "question")),
                "answer": None,
                "ok": None,
            }
            if client:
                answer, tokens = ask(client, t.text, top)
                row["answer"] = answer
                row["ok"] = t.expect in answer.lower()
                print(f"     -> ({tokens} tok) {answer}   [{'OK' if row['ok'] else 'WRONG'}]")
            results.append(row)

        stored.append(to_memory(t))  # naive write: always ADD

    # ---- summary ---------------------------------------------------------
    superseded = {2, 3, 4, 6}
    questions = {t.n for t in CONVERSATION if t.is_question}
    print("\n" + "=" * 78)
    print(f"embedder : {store_name}")
    print(
        f"final store: {len(stored)} memories -- {len(superseded)} superseded facts, "
        f"{len(questions)} stored questions, 1 restatement (t14)"
    )
    print(f"\n{'turn':>4} {'store':>5} {'rank':>5} {'crowded':>8} {'answer':>8}")
    for r in results:
        ans = "-" if r["ok"] is None else ("OK" if r["ok"] else "WRONG")
        rank = r["rank"] if r["rank"] is not None else "-"
        print(f"{r['turn']:>4} {r['size']:>5} {rank:>5} {r['crowd']:>5}/{args.k:<2} {ans:>8}")

    top1 = sum(1 for r in results if r["rank"] == 1)
    in_k = sum(1 for r in results if r["rank"] and r["rank"] <= args.k)
    print(f"\ncurrent fact ranked #1 : {top1}/{len(results)}")
    print(f"current fact in top-{args.k}  : {in_k}/{len(results)}")
    if client:
        print(f"answered correctly     : {sum(1 for r in results if r['ok'])}/{len(results)}")
    print(
        "\n'crowded' = retrieval slots spent on stale facts, decoys or stored questions."
        "\nCompare turn 5 with turn 18: same question, bigger store."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())