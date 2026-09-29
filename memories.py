"""
Toy memory corpus — 30 items.

Deliberately built with three traps that matter later in the thesis:

  TRAP A (staleness):     items 4 and 21 contradict each other; 21 is newer and correct.
  TRAP B (duplication):   items 7, 12 and 26 say nearly the same thing three ways.
  TRAP C (lexical decoy): item 15 shares vocabulary with the "editor" question but
                          answers a different question entirely.

Step 1 uses these to show what pure similarity search CAN and CANNOT do.
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class Memory:
    id: int
    text: str
    when: date

    def __repr__(self) -> str:
        return f"[{self.id:02d} {self.when}] {self.text}"


def _m(i, y, mo, d, text):
    return Memory(id=i, when=date(y, mo, d), text=text)


MEMORIES = [
    _m(1,  2026, 1, 12, "Joel is a final-year software engineering student at IIT, affiliated with the University of Westminster."),
    _m(2,  2026, 1, 12, "Joel's final year project is on memory systems for long-horizon LLM agents."),
    _m(3,  2026, 1, 14, "Joel finished a 12-month industrial placement at Docupath AI as a software engineering intern."),
    _m(4,  2026, 1, 15, "Joel's main code editor is VS Code."),                                   # TRAP A: stale
    _m(5,  2026, 1, 18, "Joel prefers explanations that avoid unexplained jargon."),
    _m(6,  2026, 1, 20, "The supervisor meeting is every second Thursday at 10am."),
    _m(7,  2026, 1, 22, "Joel drinks his coffee black, no sugar."),                                # TRAP B
    _m(8,  2026, 1, 25, "Joel's placement work included CI/CD pipelines and backend refactoring."),
    _m(9,  2026, 2, 2,  "The thesis deadline is 30 April."),
    _m(10, 2026, 2, 4,  "Joel reads papers in the evening, not the morning."),
    _m(11, 2026, 2, 6,  "Mem0 is the likely primary baseline system for the thesis evaluation."),
    _m(12, 2026, 2, 9,  "Joel takes his coffee without milk or sweetener."),                       # TRAP B
    _m(13, 2026, 2, 11, "LoCoMo is a benchmark for long-conversation memory recall."),
    _m(14, 2026, 2, 13, "Joel is based in Negombo, Sri Lanka."),
    _m(15, 2026, 2, 15, "Joel edits his thesis chapters in Overleaf because of the LaTeX support."),  # TRAP C: decoy
    _m(16, 2026, 2, 17, "MemoryArena tests whether an agent can act on memory, not just recall it."),
    _m(17, 2026, 2, 20, "Joel's laptop has 16GB of RAM, which limits local model experiments."),
    _m(18, 2026, 2, 22, "The literature review draft is due two weeks before the full thesis."),
    _m(19, 2026, 2, 25, "Joel prefers bullet-pointed notes over long prose."),
    _m(20, 2026, 3, 1,  "AgentRunbook-C takes about 108 seconds per query."),
    _m(21, 2026, 3, 3,  "Joel switched from VS Code to Neovim and now uses Neovim daily."),        # TRAP A: correct
    _m(22, 2026, 3, 5,  "Joel plays chess casually, mostly blitz."),
    _m(23, 2026, 3, 8,  "The project must be written up in the university's IEEE-style template."),
    _m(24, 2026, 3, 10, "Joel trains at the gym four times a week."),
    _m(25, 2026, 3, 12, "Embeddings turn a piece of text into a list of numbers so texts can be compared."),
    _m(26, 2026, 3, 15, "When offered milk for his coffee, Joel declines."),                        # TRAP B
    _m(27, 2026, 3, 18, "Joel's supervisor asked for a one-page problem statement before the next meeting."),
    _m(28, 2026, 3, 20, "LongMemEval-V2 scales histories up to 115 million tokens."),
    _m(29, 2026, 3, 22, "Joel wants the build and the reading to happen in parallel, not one then the other."),
    _m(30, 2026, 3, 25, "The fastest retrieval methods score around 42-58% on LongMemEval-V2."),
    _m(31, 2026, 3, 27, "AgentRunbook-C is the most accurate method tested on LongMemEval-V2."),
]


# Questions used in the Step 1 demo.
#
# `category` is the important column. It predicts WHY a question fails, and
# therefore what would fix it:
#
#   "lexical"       - the question and the answer share no words. A better
#                     embedder should fix this. Not a research problem.
#   "architectural" - the right memory is found, but the store has no way to
#                     rank it correctly. A better embedder will NOT fix this.
#                     This is the research problem.
#
# Re-run with a real semantic embedder and check: the lexical rows should flip
# to HIT. The architectural rows should not.
PROBE_QUESTIONS = [
    ("What code editor does Joel use?",              21, "architectural / staleness"),
    ("How does Joel take his coffee?",                7, "architectural / duplication"),
    ("When is the thesis due?",                       9, "control (should pass)"),
    ("How slow is the most accurate memory method?", 20, "multi-hop"),
    ("Where does Joel live?",                        14, "lexical"),
]
