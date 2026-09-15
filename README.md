# memory_lab

Hands-on build for the agent-memory FYP. One directory, no framework, nothing
hidden. Each step is a file you can run on its own and read end to end.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python step0_smoke_test.py     # environment + one LLM call
python step1_retrieval.py      # vector search over 30 toy memories
```

Step 1 runs with no API key. It falls back to TF-IDF if
`sentence-transformers` isn't installed.

## Step map

| Step | File | Status | What it proves |
|---|---|---|---|
| 0 | `step0_smoke_test.py` | done | The environment works. |
| 1 | `step1_retrieval.py` | done | Similarity search finds *relevant* memories, not *correct* ones. |
| 2 | `step2_rag.py` | next | Retrieved memories in a prompt actually change the answer. |
| 3 | `step3_agent_loop.py` | — | Memory grows unbounded across turns; retrieval quality degrades. |
| 4 | `step4_consolidation.py` | — | Compressing raw turns into durable facts between sessions. |
| 5 | `step5_similarity_scorer.py` | — | The similarity-based keep/drop baseline your thesis argues against. |
| 6 | `step6_eval.py` | — | A LoCoMo slice, scored with LLM-as-judge or F1. |
| 7 | `step7_reasoning_scorer.py` | — | Stub of the actual contribution. |

## Files

- `memories.py` — 30 toy memories plus 5 probe questions. Not random: it contains
  a contradicted fact, a triple near-duplicate, and a lexical decoy, each planted
  to make a specific failure visible.
- `embedder.py` — `TfidfEmbedder` (offline), `LocalEmbedder` (sentence-transformers),
  `VoyageEmbedder` (hosted). Same interface, swap freely.
- `step1_retrieval.py` — `MemoryStore` with `add_all` / `search`, cosine similarity,
  top-k ranking, and a scored probe harness.

## The experiment built into Step 1

The probe questions are tagged `lexical` or `architectural`. That tag is a
prediction about *what would fix the failure*:

- **lexical** — question and answer share no vocabulary. A real embedder fixes it.
- **architectural** — the right memory is retrieved but ranked below a wrong one.
  No embedder fixes it.

Run Step 1 twice: once on TF-IDF, once with `sentence-transformers` installed.
Record which rows flip. The rows that don't flip are the ones worth a thesis.

Current TF-IDF baseline: top-1 **1/5**, recall@3 **3/5**.

That gap between top-1 and recall@3 is the useful signal. The system is *finding*
more than it is *ranking correctly* — which is the same shape as the paper's
oracle-retrieval finding, where handing over perfect evidence still only got
models to 60–65%.

## What Step 1 deliberately does not have

Each of these becomes a later step, so notice its absence now:

- no forgetting, no eviction, no memory budget
- no consolidation — every raw turn is stored verbatim forever
- no supersession — nothing records that memory 21 invalidates memory 4
- no structure — one flat pool, not the separate raw / event / procedure pools
  that AgentRunbook-R uses
- no ingestion cost accounting — embedding happens at `add_all` and is never measured

## Framing note

This build plan is scoped to the **reasoning-aligned consolidation** framing
(LoCoMo + MemoryArena), which is not the same scope as the three LME-V2 directions
— cascade routing, ingestion-cost benchmarking, bi-temporal invalidation.

Steps 0–4 are foundational and survive either framing. Steps 5–7 assume the
consolidation framing and would need rewriting under a cascade or bi-temporal
thesis. That decision can wait until Step 4, not longer.
