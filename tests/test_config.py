import importlib


def test_provider_switch(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    import src.core.config as cfg
    importlib.reload(cfg)
    assert cfg.Config.LLM_PROVIDER == "openai"
    assert "llm_provider" in cfg.Config.summary()


def test_bool_parsing(monkeypatch):
    monkeypatch.setenv("USE_HYBRID", "false")
    import src.core.config as cfg
    importlib.reload(cfg)
    assert cfg.Config.USE_HYBRID is False
