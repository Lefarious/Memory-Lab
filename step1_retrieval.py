"""
STEP 1 -- Semantic retrieval over a toy memory store.

Goal: given a question, return the k memories most likely to help answer it,
without an LLM in the loop at all. Pure vector search.

Run:  python step1_retrieval.py

What to watch for in the output: three of the five probe questions are answered
correctly. Two are not, and they fail for two DIFFERENT reasons. Those two failures
are the whole reason your thesis exists -- don't skim past them.
"""

from __future__ import annotations

import numpy as np

from embedder import BaseEmbedder, get_embedder
from memories import MEMORIES, PROBE_QUESTIONS, Memory

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default="auto",
                choices=["auto", "tfidf", "local", "voyage"])
args = ap.parse_args()

embedder = get_embedder(args.embedder)

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Angle between two vectors, ignoring their length. 1.0 = same direction,
    0.0 = unrelated. Length is ignored on purpose: a long memory and a short one
    about the same thing should score the same.
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


class MemoryStore:
    """
    The smallest thing that deserves the name. Three operations:
      add()    -- write a memory, embed it now (at insertion time, not query time)
      search() -- embed the question, score every memory, return the best k
      __len__  -- how much have we accumulated

    Note what is missing, because you will add each of these in later steps:
      no forgetting, no consolidation, no notion of which memory supersedes which,
      no ingestion cost accounting, no structure beyond one flat pool.
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.items: list[Memory] = []
        self._matrix: np.ndarray | None = None

    def add_all(self, memories: list[Memory]) -> None:
        self.items.extend(memories)
        corpus = [m.text for m in self.items]
        self.embedder.fit(corpus)
        self._matrix = self.embedder.encode(corpus)

    def search(self, question: str, k: int = 3) -> list[tuple[float, Memory]]:
        if self._matrix is None:
            raise RuntimeError("Store is empty.")
        q = self.embedder.encode([question])[0]
        scores = cosine_similarity(q, self._matrix)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.items[i]) for i in top]

    def __len__(self) -> int:
        return len(self.items)


def main() -> None:
    embedder = get_embedder()
    store = MemoryStore(embedder)
    store.add_all(MEMORIES)

    print(f"embedder : {embedder.name}")
    print(f"memories : {len(store)}")
    print("=" * 78)

    hits = 0
    misses_by_category: dict[str, int] = {}
    recall_at_3 = 0

    for question, correct_id, category in PROBE_QUESTIONS:
        results = store.search(question, k=3)
        top_id = results[0][1].id
        ok = top_id == correct_id
        hits += ok
        recall_at_3 += any(m.id == correct_id for _, m in results)
        if not ok:
            misses_by_category[category] = misses_by_category.get(category, 0) + 1

        print(f"\nQ: {question}   [{category}]")
        for rank, (score, mem) in enumerate(results, 1):
            marker = "<-- correct" if mem.id == correct_id else ""
            print(f"   {rank}. {score:.3f}  {mem}  {marker}")
        print(f"   verdict: {'HIT ' if ok else 'MISS'} (expected id {correct_id}, got {top_id})")

    n = len(PROBE_QUESTIONS)
    print("\n" + "=" * 78)
    print(f"top-1 accuracy : {hits}/{n}")
    print(f"recall@3       : {recall_at_3}/{n}   <- how often the answer was retrieved at all")
    if misses_by_category:
        print("misses by cause:")
        for cat, count in sorted(misses_by_category.items()):
            print(f"   {count} x {cat}")
    print(
        "\nRead the two columns separately.\n"
        "  LEXICAL misses mean the question and its answer share no words. TF-IDF\n"
        "  cannot see past that. Swap in a real semantic embedder and these flip to\n"
        "  HIT. Engineering problem, already solved by other people.\n\n"
        "  ARCHITECTURAL misses are different. The right memory IS in the top 3 --\n"
        "  recall@3 catches it -- but the store cannot rank it first. In one case two\n"
        "  memories contradict and nothing encodes which one superseded the other. In\n"
        "  the other, three near-duplicates crowd the budget so a single fact costs\n"
        "  three slots. A better embedder does not touch either. That gap is the thesis."
    )


if __name__ == "__main__":
    main()
