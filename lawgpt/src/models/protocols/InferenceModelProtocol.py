from typing import Protocol


class InferenceModelProtocol(Protocol):
    def ask(self, question: str) -> str:
        ...

    def load_model(self) -> None:
        ...
