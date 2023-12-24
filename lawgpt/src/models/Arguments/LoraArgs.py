from dataclasses import dataclass, field
from typing import List, Union


def default_target_modules():
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ]


@dataclass
class LoraArgs:
    r: int = field(default=8, metadata={"help": "Lora attention dimension."})

    target_modules: List[str] = field(
        default_factory=default_target_modules,
        metadata={"help": "The names of the modules to apply Lora to."},
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "The alpha parameter for Lora scaling."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "The dropout probability for Lora layers."}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses Conv1D which stores weights like (fan_in, fan_out) and hence this should be set to True."
        },
    )
    bias: str = field(
        default="none",
        metadata={
            "help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation."
        },
    )
    # modules_to_save: List[str] = field(
    #     metadata={
    #         "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
    #     }
    # )
    # layers_to_transform: Union[List[int], int] = field(
    #     metadata={
    #         "help": "The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA transformations on the layer at this index."
    #     }
    # )
    # layers_pattern: str = field(
    #     metadata={
    #         "help": "The layer pattern name, used only if layers_to_transform is different from None and if the layer pattern is not in the common layers pattern."
    #     }
    # )
    # rank_pattern: dict = field(
    #     metadata={
    #         "help": "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by r."
    #     }
    # )
    # alpha_pattern: dict = field(
    #     metadata={
    #         "help": "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha."
    #     }
    # )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task type of the model."}
    )
