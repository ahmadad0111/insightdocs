"""Short conversational memory that stores both question and answer.

The previous version only kept questions; this keeps full turns and
renders them for prompt context and for query expansion.
"""
from src.core.config import Config


class ConversationMemory:
    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or Config.MAX_HISTORY_TURNS
        self.turns = []  # list of {"question": str, "answer": str}

    def add(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        self.turns = self.turns[-self.max_turns:]

    def get_context(self) -> str:
        if not self.turns:
            return ""
        blocks = []
        for turn in self.turns:
            blocks.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        return "\n\n".join(blocks)

    def last_questions(self, n: int = 2) -> str:
        return "\n".join(t["question"] for t in self.turns[-n:])

    def clear(self) -> None:
        self.turns = []
