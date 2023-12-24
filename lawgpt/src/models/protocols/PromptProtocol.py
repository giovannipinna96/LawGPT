from typing import Protocol


class PromptProtocol(Protocol):
    def create_prompt(self) -> str:
        ...
