class DummyLLM:

    def __call__(self, prompt):
        return "LLM RESPONSE PLACEHOLDER (we will connect real model next)"

from openai import OpenAI


class OpenAILLM:

    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers using provided context only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content


import requests
import json

from src.core.config import Config

class LocalLLM:

    def __init__(self, model=None):
        self.model = model or Config.LLM_MODEL
        self.url = Config.OLLAMA_URL

    def __call__(self, prompt):

        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]


    def stream(self, prompt):

        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))

                token = data.get("response", "")

                if token:
                    yield token