import sys

sys.path.insert(1, "../")

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)

from Arguments.ModelArgs import ModelArgs
from Arguments.BitsAndBytesArgs import BitsAndBytesArgs
from Arguments.LoraArgs import LoraArgs
from Arguments.DataTrainingArgs import DataTrainingArgs


def parse_arguments():
    parser = HfArgumentParser(
        (
            TrainingArguments,
            BitsAndBytesArgs,
            ModelArgs,
            LoraArgs,
            DataTrainingArgs
            # DataCollatorForSeq2Seq,
            # DataCollatorForLanguageModeling
        )
    )
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    train_args, bits_args, model_args, lora_args, data_args = parse_arguments()
