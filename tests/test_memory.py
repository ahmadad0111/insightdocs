from src.rag.memory.conversation_memory import ConversationMemory


def test_memory_stores_full_turns_and_truncates():
    mem = ConversationMemory(max_turns=2)
    mem.add("q1", "a1")
    mem.add("q2", "a2")
    mem.add("q3", "a3")
    ctx = mem.get_context()
    assert "q1" not in ctx          # evicted
    assert "q3" in ctx and "a3" in ctx
    assert len(mem.turns) == 2


def test_memory_clear():
    mem = ConversationMemory()
    mem.add("q", "a")
    mem.clear()
    assert mem.get_context() == ""
