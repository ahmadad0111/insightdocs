from src.rag.agentic.router import heuristic_route, QueryRouter


def test_greetings_and_meta_route_direct():
    assert heuristic_route("hi") == "direct"
    assert heuristic_route("Thanks!") == "direct"
    assert heuristic_route("what can you do?") == "direct"


def test_real_questions_route_retrieve():
    assert heuristic_route("What is federated learning?") == "retrieve"
    assert heuristic_route("Summarize the results section") == "retrieve"
    assert heuristic_route("") == "direct"


def test_confident_greeting_short_circuits_without_llm():
    calls = []
    def spy(p): calls.append(p); return "RETRIEVE"
    r = QueryRouter(llm=spy)
    assert r.route("hi") == "direct"      # heuristic wins
    assert calls == []                     # LLM never consulted


def test_llm_decides_ambiguous_then_falls_back():
    # a real question: LLM can downgrade to direct
    assert QueryRouter(llm=lambda p: "DIRECT").route("tell me about yourself in detail") == "direct"
    # LLM garbage on a real question -> defaults to retrieve
    assert QueryRouter(llm=lambda p: "banana").route("What is the main contribution?") == "retrieve"
    # LLM raises on a real question -> retrieve
    def boom(p): raise RuntimeError("down")
    assert QueryRouter(llm=boom).route("What is the method?") == "retrieve"
