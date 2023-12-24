import os
import openai
from models_llama.AbstractLanguageModel import AbstractLargeLanguageModel


class ChatGPT(AbstractLargeLanguageModel):
    def __init__(self) -> None:
        super().__init__("ChatGPT")

    def ask(self, question: str) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self._INTRODUCTION_TO_QUESTION + question}
            ],
        )
        res = completion.choices[0].message.content
        return res

    def _load_model(self) -> None:
        openai.api_key = "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"  # os.getenv("OPENAI_API_KEY")
