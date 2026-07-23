"""Provider-switchable LLM layer.

All providers expose the same interface:
    __call__(prompt: str) -> str          (blocking)
    stream(prompt: str) -> Iterator[str]   (token stream)

Choose the provider with LLM_PROVIDER=ollama|openai|anthropic.
"""
import json
import requests

from src.core.config import Config

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided context."
)


class OllamaLLM:
    def __init__(self, model: str = None, url: str = None):
        self.model = model or Config.OLLAMA_MODEL
        self.url = url or Config.OLLAMA_URL

    def __call__(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": Config.LLM_TEMPERATURE}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def stream(self, prompt: str):
        resp = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True,
                  "options": {"temperature": Config.LLM_TEMPERATURE}},
            stream=True, timeout=120,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            token = json.loads(line.decode("utf-8")).get("response", "")
            if token:
                yield token


class OpenAILLM:
    def __init__(self, model: str = None, api_key: str = None):
        from openai import OpenAI
        self.model = model or Config.OPENAI_MODEL
        self.client = OpenAI(api_key=api_key or Config.OPENAI_API_KEY)

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt),
            temperature=Config.LLM_TEMPERATURE,
        )
        return resp.choices[0].message.content

    def stream(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt),
            temperature=Config.LLM_TEMPERATURE, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLM:
    def __init__(self, model: str = None, api_key: str = None):
        import anthropic
        self.model = model or Config.ANTHROPIC_MODEL
        self.client = anthropic.Anthropic(api_key=api_key or Config.ANTHROPIC_API_KEY)

    def __call__(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
            temperature=Config.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def stream(self, prompt: str):
        with self.client.messages.stream(
            model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
            temperature=Config.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


def get_llm(provider: str = None):
    provider = (provider or Config.LLM_PROVIDER).lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use ollama|openai|anthropic)")
