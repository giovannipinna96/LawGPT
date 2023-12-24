import os
import openai
from models_llama.AbstractLanguageModel import AbstractLargeLanguageModel


class GPT4(AbstractLargeLanguageModel):
    def __init__(self) -> None:
        super().__init__("GPT4")

    def ask(self, question: str) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": self._INTRODUCTION_TO_QUESTION + question}
            ],
        )
        return completion.choices[0].message.content

    def _load_model(self) -> None:
        openai.api_key = "your key"  # "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"  # os.getenv("OPENAI_API_KEY")
