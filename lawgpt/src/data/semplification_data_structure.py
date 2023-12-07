# semplification data structure

from typing import Protocol


class SemplificationDataStructure(Protocol):
    def get_semply_text(self):
        ...

    def get_original_text(self):
        ...
