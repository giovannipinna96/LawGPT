from typing import Protocol


class MakeDataset(Protocol):
    @property
    def name(self):
        ...

    def create_data(self):
        ...
