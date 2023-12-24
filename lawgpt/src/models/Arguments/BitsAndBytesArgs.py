from dataclasses import dataclass, field

import torch


@dataclass
class BitsAndBytesArgs:
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load the model in 4-bit format"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "4-bit quantization type"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float32", metadata={"help": "4-bit compute dtype"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "Use double quantization in 4-bit quantization"},
    )


def compute_dtype(bnb_4bit_compute_dtype: str):
    return getattr(torch, bnb_4bit_compute_dtype)
