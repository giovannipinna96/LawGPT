# summarization data structure

from typing import Protocol


class SummarizationDataStructure(Protocol):
    def get_summary_text(self):
        ...

    def get_original_text(self):
        ...
