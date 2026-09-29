"""
STEP 2 -- Minimal RAG. Put retrieved memories into a prompt and watch the answer change.

Step 1 gave you a ranking number. A number is easy to shrug off. This step converts
that same failure into something you cannot shrug off: a fluent, confident, wrong
sentence.

Every probe question is asked three ways:

  A. CLOSED BOOK   no memories at all. The model should say it doesn't know.
                   If it answers anyway, that is hallucination, and it means the
                   question was answerable from general knowledge -- a bad probe.

  B. RETRIEVED     top-k from your Step 1 store. This is a real memory system.

  C. ORACLE        the one correct memory, handed over directly, no search at all.
                   This is the ceiling. It removes the retrieval problem entirely
                   and asks: even with perfect evidence, does the model get it right?

Arm C matters more than it looks. The LongMemEval-V2 paper found that oracle access
still only reached 60-65% raw accuracy -- meaning a large slice of the difficulty is
reasoning over evidence, not finding it. Arm C is how you measure that split in your
own system.

Run:  python step2_rag.py --dry-run          # prints the prompts, no API call, no cost
      python step2_rag.py                    # calls the model (k=3)
      python step2_rag.py -k 1               # only the top memory reaches the model
      python step2_rag.py --embedder tfidf   # force a specific embedder
"""

from __future__ import annotations

import argparse
import os
import sys

# Load ANTHROPIC_API_KEY from a .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from embedder import get_embedder
from memories import MEMORIES, PROBE_QUESTIONS, Memory
from step1_retrieval import MemoryStore

MODEL = os.environ.get("MEMORY_LAB_MODEL", "claude-haiku-4-5-20251001")

SYSTEM = (
    "You answer questions about Joel using only the memories provided. "
    "Each memory is timestamped. If the memories do not contain the answer, "
    "say exactly: I don't know. Do not guess. Keep the answer to one short sentence."
)


def build_prompt(question: str, memories: list[Memory]) -> str:
    if not memories:
        block = "(no memories available)"
    else:
        block = "\n".join(f"- [{m.when}] {m.text}" for m in memories)
    return f"Memories:\n{block}\n\nQuestion: {question}"


def ask(client, question: str, memories: list[Memory]) -> tuple[str, int]:
    prompt = build_prompt(question, memories)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.usage.input_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print prompts, make no API calls")
    ap.add_argument("-k", type=int, default=3, help="how many memories to retrieve")
    ap.add_argument(
        "--embedder",
        default=None,
        help="force an embedder (e.g. tfidf, local); default picks the best installed",
    )
    args = ap.parse_args()

    embedder = get_embedder(args.embedder) if args.embedder else get_embedder()
    store = MemoryStore(embedder)
    store.add_all(MEMORIES)
    by_id = {m.id: m for m in MEMORIES}

    client = None
    if not args.dry_run:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ANTHROPIC_API_KEY not set. Use --dry-run to inspect prompts for free.")
            return 1
        import anthropic

        client = anthropic.Anthropic(api_key=key)

    print(f"embedder : {store.embedder.name}")
    print(f"model    : {MODEL if not args.dry_run else '(dry run)'}")
    print(f"k        : {args.k}")
    print("=" * 78)

    total_tokens = 0

    for question, correct_id, category in PROBE_QUESTIONS:
        retrieved = [m for _, m in store.search(question, k=args.k)]
        oracle = [by_id[correct_id]]

        arms = [
            ("A closed book", []),
            ("B retrieved  ", retrieved),
            ("C oracle     ", oracle),
        ]

        print(f"\nQ: {question}   [{category}]")
        print(f"   ground truth: {by_id[correct_id].text}")

        for label, mems in arms:
            if args.dry_run:
                print(f"\n   --- {label.strip()} prompt ---")
                for line in build_prompt(question, mems).splitlines():
                    print(f"   | {line}")
            else:
                answer, tokens = ask(client, question, mems)
                total_tokens += tokens
                print(f"   {label} ({tokens:>4} tok): {answer}")

    print("\n" + "=" * 78)
    if not args.dry_run:
        print(f"total input tokens: {total_tokens}")
        print(
            "\nWhat to record for each question:\n"
            "  - Did A correctly refuse? If it answered, drop that probe. It was\n"
            "    answerable without memory, so it measures nothing.\n"
            "  - Did B answer correctly, wrongly, or refuse? A CONFIDENT WRONG answer\n"
            "    is the result you want to capture. It is the staleness failure made\n"
            "    visible, and it is far more damaging than a refusal.\n"
            "  - Did C answer correctly? If C is wrong too, retrieval was never the\n"
            "    problem for that question -- the model cannot reason over the evidence\n"
            "    even when handed it. That is a separate, harder finding.\n\n"
            "The gap between B and C is the value a better memory system could add.\n"
            "The gap between C and 100% is the ceiling no memory system can lift."
        )
    else:
        print("Dry run complete. No API calls made, nothing billed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())