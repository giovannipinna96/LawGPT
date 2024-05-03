# summarization data structure

from typing import Protocol


class SummarizationDataStructure(Protocol):
    def get_summary_text(self):
        """
        Get the summary text for the given input.

        Returns:
            str: The summary text.
        """
        ...

    def get_original_text(self):
        """
        Get the original text for the given input.

        Returns:
            str: The original text.
        """
        ...
