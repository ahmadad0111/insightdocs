"""Query decomposition.

A complex question ("What is FL and how does it reduce communication cost?")
retrieves better when split into focused sub-questions, each searched
separately, with the union of results used as context. Simple questions are
returned unchanged so we never add latency for no reason.

The LLM does the split when available; a heuristic decides whether a split is
even worth attempting and provides a fallback. Pure helpers are unit-tested.
"""
import re

from src.core.config import Config
from src.core.logging import logger

_CONJUNCTIONS = re.compile(r"\b(and|also|as well as|additionally|plus)\b", re.IGNORECASE)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference between|"
                      r"differ|both)\b", re.IGNORECASE)

DECOMPOSE_PROMPT = """Break the user's question into the minimal set of
self-contained sub-questions needed to answer it fully. If it is already a
single simple question, return it unchanged.

Rules:
- One sub-question per line.
- No numbering, no bullets, no commentary.
- At most {max_sub} lines.

Question: {query}

Sub-questions:"""


def needs_decomposition(query: str) -> bool:
    """Cheap check: is this query likely multi-part?"""
    q = (query or "").strip()
    if len(q.split()) < 6:
        return False
    if q.count("?") > 1:
        return True
    if _COMPARE.search(q):
        return True
    # a conjunction joining two question-like clauses
    if _CONJUNCTIONS.search(q) and len(q.split()) >= 10:
        return True
    return False


def parse_subquestions(text: str, max_sub: int = 4):
    """Parse an LLM response into a clean list of sub-questions."""
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip leading numbering / bullets like "1.", "1)", "- ", "* "
        line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if line:
            lines.append(line)
    # de-duplicate while preserving order
    seen, out = set(), []
    for l in lines:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out[:max_sub]


class QueryDecomposer:
    def __init__(self, llm=None, max_sub: int = None):
        self.llm = llm
        self.max_sub = max_sub or Config.MAX_SUBQUESTIONS

    def decompose(self, query: str):
        """Return a list of sub-questions (>=1). Always includes something usable."""
        if not needs_decomposition(query) or self.llm is None:
            return [query]
        try:
            raw = self.llm(DECOMPOSE_PROMPT.format(query=query, max_sub=self.max_sub))
            subs = parse_subquestions(raw, max_sub=self.max_sub)
            if len(subs) >= 2:
                logger.info(f"Decomposed into {len(subs)} sub-questions")
                return subs
        except Exception as exc:
            logger.warning(f"Decomposer LLM failed, using original query: {exc}")
        return [query]
