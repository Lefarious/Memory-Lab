"""
Shared engine for Steps 4-7. Not run directly -- the step scripts import it.

The whole thesis question lives in one function signature:

    policy.consolidate(kept, session, date) -> new kept list

At the end of every session the agent holds two piles: what it KEPT from before,
and the RAW turns of the session that just ended. A policy decides what survives.
Everything else in this file is plumbing that stays identical across policies,
so any difference in the results is caused by the policy alone.

Policies:
  raw         never consolidate -- the Step 3 baseline, store grows forever
  recency     keep the newest N items (sanity baseline: "is anything smarter than this?")
  llm         Step 4: an LLM summarises everything into <= N durable facts
  similarity  Step 5: keep the N items most similar to recent context
  ablation    Step 7: keep the N items whose removal breaks the most probe answers
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from memories import Memory, _m
from step1_retrieval import MemoryStore

MODEL = os.environ.get("MEMORY_LAB_MODEL", "claude-haiku-4-5-20251001")

TOY_SYSTEM = (
    "You are Joel's assistant. Answer using only the memories provided. "
    "Each memory is timestamped; when memories conflict, prefer the most recent. "
    "If the memories do not contain the answer, say exactly: I don't know. "
    "Do not guess. Keep the answer to one short sentence."
)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Item:
    """A memory plus the date we tracked for it (kept separately so we never
    depend on how Memory stores its timestamp)."""
    mem: Memory
    date: tuple[int, int, int]

    @property
    def text(self) -> str:
        return self.mem.text


@dataclass
class Question:
    text: str
    grade: Callable[[str], float]          # answer -> score in [0, 1]
    expect: str | None = None              # keyword for retrieval-level hit (toy only)
    label: str = ""                        # e.g. "t11" or "cat 2"
    gold: str = ""


@dataclass
class Event:
    kind: str                              # "turn" | "question" | "end_session"
    item: Item | None = None
    question: Question | None = None
    date: tuple[int, int, int] | None = None


class IdGen:
    """Fresh ids for consolidated facts, clear of raw-turn ids."""
    def __init__(self, start: int = 10_000):
        self.n = start

    def __call__(self) -> int:
        self.n += 1
        return self.n


def keyword_in(keyword: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) is not None


def strip_json(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


# --------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------

def make_client(dry_run: bool):
    if dry_run:
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Use --dry-run to run without the model.")
    import anthropic
    return anthropic.Anthropic(api_key=key)


def call(client, system: str, prompt: str, max_tokens: int = 300) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def format_items(items: list[Item]) -> str:
    if not items:
        return "(no memories available)"
    return "\n".join(
        f"- [{it.date[0]:04d}-{it.date[1]:02d}-{it.date[2]:02d}] {it.text}" for it in items
    )


def answer(client, system: str, question: str, items: list[Item]) -> str:
    prompt = f"Memories:\n{format_items(items)}\n\nQuestion: {question}"
    return call(client, system, prompt, max_tokens=150)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

class Policy:
    name = "base"
    window = 5                             # recent turns that count as "context" (similarity)

    def __init__(self, n: int, embedder, client, ids: IdGen, system: str = TOY_SYSTEM,
                 subject: str = "Joel"):
        self.n, self.embedder, self.client, self.ids, self.system = n, embedder, client, ids, system
        self.subject = subject             # who the memory is about, used in LLM prompts
        self.note = ""                     # shown once if the policy fell back

    def consolidate(self, kept: list[Item], session: list[Item], date) -> list[Item]:
        raise NotImplementedError


class RawPolicy(Policy):
    name = "raw"

    def consolidate(self, kept, session, date):
        return kept + session


class RecencyPolicy(Policy):
    name = "recency"

    def consolidate(self, kept, session, date):
        return (kept + session)[-self.n:]


class SimilarityPolicy(Policy):
    """Step 5. Score each candidate by cosine similarity to recent context and keep
    the top N. This is the baseline the thesis argues against: it keeps what looks
    RELATED to recent talk, not what is USEFUL for future answers."""
    name = "similarity"

    def consolidate(self, kept, session, date):
        candidates = kept + session
        if len(candidates) <= self.n:
            return candidates
        context = " ".join(it.text for it in session[-self.window:]) or candidates[-1].text
        store = MemoryStore(self.embedder)
        store.add_all([it.mem for it in candidates])
        ranked = store.search(context, k=len(candidates))
        keep_ids = {m.id for _, m in ranked[: self.n]}
        return [it for it in candidates if it.mem.id in keep_ids]   # keep time order


class LLMPolicy(Policy):
    """Step 4. Hand everything to an LLM and ask for <= N durable facts. The LLM
    decides what to keep by its own rules of thumb -- nobody measures usefulness."""
    name = "llm"

    PROMPT = (
        "You maintain a long-term memory about {subject}.\n\n"
        "EXISTING FACTS:\n{kept}\n\nNEW CONVERSATION TURNS:\n{session}\n\n"
        "Rewrite the memory as at most {n} short, durable facts. Rules:\n"
        "- When a new turn contradicts an old fact, keep only the newer version.\n"
        "- Drop questions, chit-chat and duplicates.\n"
        "- Each fact must be one self-contained sentence.\n"
        'Return ONLY a JSON list of strings, e.g. ["First fact.", "Second fact."]'
    )

    def consolidate(self, kept, session, date):
        if self.client is None:
            self.note = "dry run: llm policy fell back to recency (no model calls)"
            return RecencyPolicy.consolidate(self, kept, session, date)
        prompt = self.PROMPT.format(kept=format_items(kept), session=format_items(session),
                                    n=self.n, subject=self.subject)
        raw = call(self.client, "You return only valid JSON.", prompt, max_tokens=600)
        try:
            facts = [str(f) for f in json.loads(strip_json(raw))][: self.n]
        except (json.JSONDecodeError, TypeError):
            self.note = "llm returned invalid JSON once; kept previous facts for that session"
            return kept
        y, m, d = date
        return [Item(_m(self.ids(), y, m, d, f), date) for f in facts]


class AblationPolicy(Policy):
    """Step 7 (stub of the thesis contribution).

    1. Ask the LLM for a few probe questions (with short answers) that the
       candidate memories should be able to answer, using the newest information.
    2. For each candidate, remove it and re-ask every probe.
    3. utility(item) = number of probes that were right WITH it and wrong WITHOUT it.
    4. Keep the N highest-utility items; break ties by recency.

    Known weakness, worth a paragraph in the thesis: leave-one-out gives ZERO
    utility to both copies of a duplicated fact, because each copy covers for the
    other. That is why ties matter so much here."""
    name = "ablation"
    probes_per_session = 4

    PROBE_PROMPT = (
        "Here are memories about {subject}:\n{items}\n\n"
        "Write {p} short questions a user might later ask that these memories answer. "
        "Each answer must be 1-3 words and reflect the MOST RECENT information. "
        'Return ONLY JSON: [{{"q": "...", "a": "..."}}]'
    )

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0
        self.last_utility: list[tuple[int, Item]] = []

    def _correct(self, probe, items) -> bool:
        self.calls += 1
        return keyword_in(probe["a"], answer(self.client, self.system, probe["q"], items))

    def consolidate(self, kept, session, date):
        candidates = kept + session
        if len(candidates) <= self.n:
            return candidates
        if self.client is None:
            self.note = "dry run: ablation policy fell back to similarity (no model calls)"
            return SimilarityPolicy.consolidate(self, kept, session, date)

        raw = call(self.client, "You return only valid JSON.",
                   self.PROBE_PROMPT.format(items=format_items(candidates), p=self.probes_per_session,
                                            subject=self.subject),
                   max_tokens=500)
        try:
            probes = [p for p in json.loads(strip_json(raw)) if p.get("q") and p.get("a")]
        except (json.JSONDecodeError, TypeError, AttributeError):
            probes = []
        # Only probes the full set gets right can be "broken" by a removal.
        probes = [p for p in probes if self._correct(p, candidates)]
        if not probes:
            self.note = "no usable probes in a session; fell back to recency there"
            return candidates[-self.n:]

        scored = []
        for idx, it in enumerate(candidates):
            without = candidates[:idx] + candidates[idx + 1:]
            utility = sum(1 for p in probes if not self._correct(p, without))
            scored.append((utility, idx, it))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)   # utility, then newest
        self.last_utility = [(u, it) for u, _, it in scored]
        keep_ids = {it.mem.id for _, _, it in scored[: self.n]}
        return [it for it in candidates if it.mem.id in keep_ids]


POLICIES = {p.name: p for p in (RawPolicy, RecencyPolicy, LLMPolicy, SimilarityPolicy, AblationPolicy)}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class Retriever:
    """Rebuilds the vector store only when the pool actually changed."""
    def __init__(self, embedder):
        self.embedder, self.key, self.store = embedder, None, None

    def top_k(self, pool: list[Item], query: str, k: int) -> list[Item]:
        if not pool:
            return []
        key = tuple(it.mem.id for it in pool)
        if key != self.key:
            self.store = MemoryStore(self.embedder)
            self.store.add_all([it.mem for it in pool])
            self.key = key
        by_id = {it.mem.id: it for it in pool}
        return [by_id[m.id] for _, m in self.store.search(query, k=min(k, len(pool)))]


def run(events: list[Event], policy: Policy, k: int, verbose: bool = True) -> list[dict]:
    kept: list[Item] = []
    session: list[Item] = []
    retriever = Retriever(policy.embedder)
    rows = []

    for ev in events:
        if ev.kind == "turn":
            session.append(ev.item)

        elif ev.kind == "end_session":
            kept = policy.consolidate(kept, session, ev.date)
            session = []
            if verbose and policy.name != "raw":
                print(f"\n  [{policy.name}] kept after session ({len(kept)} items):")
                for it in kept:
                    print(f"     - {it.text}")

        elif ev.kind == "question":
            q = ev.question
            pool = kept + session
            top = retriever.top_k(pool, q.text, k)
            hit = None if q.expect is None else any(keyword_in(q.expect, it.text) for it in top)
            ans, score = None, None
            if policy.client is not None:
                ans = answer(policy.client, policy.system, q.text, top)
                score = q.grade(ans)
            rows.append({"label": q.label, "q": q.text, "pool": len(pool),
                         "hit": hit, "answer": ans, "score": score, "gold": q.gold})
            if verbose:
                h = "" if hit is None else f" hit={'Y' if hit else 'N'}"
                s = "" if score is None else f" score={score:.2f}"
                print(f"\n  [{policy.name}] {q.label} {q.text}  (pool={len(pool)}){h}{s}")
                if ans is not None:
                    print(f"     -> {ans}")
    if policy.note:
        print(f"\n  note: {policy.note}")
    return rows


def compare(results: dict[str, list[dict]]) -> None:
    """Side-by-side table: one row per question, one column per policy."""
    names = list(results)
    first = results[names[0]]
    print("\n" + "=" * 78)
    head = f"{'question':<34}" + "".join(f"{n:>13}" for n in names)
    print(head)
    print("-" * len(head))
    for i, row in enumerate(first):
        cells = []
        for n in names:
            r = results[n][i]
            hit = "-" if r["hit"] is None else ("Y" if r["hit"] else "N")
            sc = "-" if r["score"] is None else f"{r['score']:.2f}"
            cells.append(f"{'hit ' + hit + ' ' + sc:>13}")
        print(f"{(row['label'] + ' ' + row['q'])[:33]:<34}" + "".join(cells))
    print("-" * len(head))
    tot = []
    for n in names:
        rs = results[n]
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        scores = [r["score"] for r in rs if r["score"] is not None]
        h = f"{sum(hits)}/{len(hits)}" if hits else "-"
        s = f"{sum(scores) / len(scores):.2f}" if scores else "-"
        tot.append(f"{h + ' ' + s:>13}")
    print(f"{'TOTAL (hits, mean score)':<34}" + "".join(tot))
    print("hit = expected keyword appears in the top-k retrieved memories; score = answer graded.")


# --------------------------------------------------------------------------
# Toy conversation from Step 3, split into sessions
# --------------------------------------------------------------------------

TOY_SESSION_ENDS = (6, 10, 14)   # sessions: 1-6, 7-10, 11-14, 15-19


def toy_events(session_ends=TOY_SESSION_ENDS) -> list[Event]:
    from step3_agent_loop import CONVERSATION, ID_OFFSET

    events = []
    for t in CONVERSATION:
        date = (2026, 9, t.n)
        if t.is_question:
            exp = t.expect
            events.append(Event("question", question=Question(
                text=t.text, expect=exp, label=f"t{t.n}", gold=exp,
                grade=lambda a, exp=exp: 1.0 if keyword_in(exp, a) else 0.0)))
        # Every turn -- questions included -- is stored, exactly as in Step 3.
        events.append(Event("turn", item=Item(_m(ID_OFFSET + t.n, *date, t.text), date)))
        if t.n in session_ends:
            events.append(Event("end_session", date=date))
    return events


# --------------------------------------------------------------------------
# Shared command-line setup for the step scripts
# --------------------------------------------------------------------------

def add_common_args(ap) -> None:
    ap.add_argument("--dry-run", action="store_true", help="no model calls; retrieval only")
    ap.add_argument("-k", type=int, default=3, help="memories retrieved per question")
    ap.add_argument("-n", "--max-facts", type=int, default=5, help="items kept per consolidation")
    ap.add_argument("--embedder", default=None, help="force an embedder (e.g. tfidf, local)")
    ap.add_argument("--quiet", action="store_true", help="only print the comparison table")


def setup(args):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from embedder import get_embedder
    embedder = get_embedder(args.embedder) if args.embedder else get_embedder()
    client = make_client(args.dry_run)
    return embedder, client


def run_policies(names, events, args, embedder, client, system=TOY_SYSTEM,
                 subject="Joel") -> dict:
    results = {}
    for name in names:
        print("\n" + "#" * 78 + f"\n# policy: {name}\n" + "#" * 78)
        policy = POLICIES[name](args.max_facts, embedder, client, IdGen(), system, subject)
        results[name] = run(events, policy, args.k, verbose=not args.quiet)
        if isinstance(policy, AblationPolicy) and policy.calls:
            print(f"\n  ablation used {policy.calls} answer calls")
    return results
