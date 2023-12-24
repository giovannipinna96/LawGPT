from typing import Protocol


class TrainingModelProtocol(Protocol):
    def train(self) -> None:
        ...

    def load_model(self) -> None:
        ...
