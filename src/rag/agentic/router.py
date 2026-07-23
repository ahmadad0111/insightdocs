"""Query router: decide whether a query needs document retrieval.

Two paths:
    - "retrieve": the question is about document content -> run the RAG pipeline.
    - "direct":   greeting, thanks, meta ("what can you do?"), or small talk
                  -> answer directly without retrieval (saves a round trip and
                  avoids forcing irrelevant context into the prompt).

An LLM makes the call when available; a fast, deterministic heuristic is used
as a fallback and for tests. The heuristic is intentionally conservative:
when unsure it chooses "retrieve" so we never skip grounding a real question.
"""
import re

from src.core.logging import logger

_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening)|thanks|thank you|"
    r"cheers|bye|goodbye|ok|okay|cool|nice)\b[\s!.]*$",
    re.IGNORECASE,
)
_META = re.compile(
    r"\b(who are you|what can you do|what are you|how do you work|help me|"
    r"what is your name|are you (a )?(bot|ai))\b",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Classify the user's message into exactly one label.

RETRIEVE - the message asks about the content of the user's documents,
           facts, definitions, comparisons, or anything that needs looking up.
DIRECT   - a greeting, thanks, small talk, or a meta question about you
           (the assistant) that needs no document lookup.

Reply with only one word: RETRIEVE or DIRECT.

Message: {query}
Label:"""


def heuristic_route(query: str) -> str:
    """Pure, dependency-free routing. Returns 'retrieve' or 'direct'."""
    q = (query or "").strip()
    if not q:
        return "direct"
    if _GREETING.match(q):
        return "direct"
    if _META.search(q) and "document" not in q.lower():
        return "direct"
    return "retrieve"


class QueryRouter:
    def __init__(self, llm=None, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def route(self, query: str) -> str:
        # A confident heuristic (clear greeting / meta) short-circuits: it is
        # both faster and more robust than trusting the LLM on the obvious cases.
        if heuristic_route(query) == "direct":
            return "direct"
        # Otherwise let the LLM decide; it can still downgrade to "direct".
        if self.use_llm:
            try:
                raw = self.llm(ROUTER_PROMPT.format(query=query)).strip().upper()
                if "DIRECT" in raw:
                    return "direct"
                if "RETRIEVE" in raw:
                    return "retrieve"
            except Exception as exc:
                logger.warning(f"Router LLM failed, using heuristic: {exc}")
        return "retrieve"
