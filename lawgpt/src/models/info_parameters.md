# Information about usefull parameters

## load data

I jsons file are loaded directly with `load_dataset()` an example is

```python
from dataset import load_dataset
dataset = load_dataset("json", data_files="my_file.json) 
```


For the Parquet files:
```python
from datasets import load_dataset
dataset = load_dataset("parquet",
                        data_files={'train': 'train.parquet',
                                    'test': 'test.parquet'
                                    }
                    )
```

If i have a list of dict we can use:
```python
from datasets import Dataset
my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
dataset = Dataset.from_list(my_list)
```

This is similar for pandas. 
```python
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
```

## Supervised Fine-tning Trainer (SFT for shot)

- Make sure to pass a correct value for `max_seq_length` as the default value will be set to `min(tokenizer.model_max_length, 1024)`.

- If you want to use your own prompt you can do something like (here is use th Alpaca prompt):

```python
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()
```

- **Paking dataset** :  SFTTrainer supports example packing, where multiple short examples are packed in the same input sequence to increase training efficiency. This is done with the `ConstantLengthDataset` utility class that returns constant length chunks of tokens from a stream of examples. To enable the usage of this dataset class, simply pass `packing=True` to the SFTTrainer constructor.
Note that if you use a packed dataset and if you pass `max_steps` in the training arguments you will probably train your models for more than few epochs, depending on the way you have configured the packed dataset and the training protocol. Double check that you know and understand what you are doing.

- **Control over the pretrainde model** : ypu can directly pass the kwargs of the `from_pretrained()` method to th SFTTrainer. (`torch_dtype=torch.bfloat16`)

- **Training adapters (with PEFT)** : The function also adapt the PEFT library.
```python
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset = load_dataset("imdb", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    "EleutherAI/gpt-neo-125m",
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config             
)

trainer.train()
```

Note that in case of training adapters, we manually add a saving callback to automatically save the adapters only.

```python
class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir,
                                    f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
```

If you want to add more callbacks, make sure to add this one as well to properly save the adapters only during training.

```python
callbacks = [YourCustomCallback(), PeftSavingCallback()]

trainer = SFTTrainer(
    "EleutherAI/gpt-neo-125m",
    train_dataset=dataset,
    dataset_text_field="text",
    torch_dtype=torch.bfloat16,
    peft_config=peft_config,
    callbacks=callbacks
)
```

- Load the models in 8 bit is also possible with SFTTrainer. The model that you pass to the trainer must be loaded in 8 bit.
```python
...
peft_config = LoraConfig(...)

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-125m",
    load_in_8bit=True,
    device_map="auto"
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    torch_dtype=torch.bfloat16,
    peft_config=peft_config
)

trainer.train()
```

## Best practices

Pay attention to the following best practices when training a model with that trainer:

- SFTTrainer always pads by default the sequences to the `max_seq_length` argument of the SFTTrainer. If none is passed, the trainer will retrieve that value from the tokenizer. Some tokenizers do not provide default value, so there is a check to retrieve **the minimum between 2048 and that value.** Make sure to check it before training.
- For training adapters in 8bit, you might need to tweak the arguments of the `prepare_model_for_int8_training` method from PEFT, hence we advise users to use `prepare_in_int8_kwargs` field, or create the PeftModel outside the SFTTrainer and pass it.
- For a more memory-efficient training using adapters, you can load the base model in 8bit, for that simply add `load_in_8bit` argument when creating the SFTTrainer, or create a base model in 8bit outside the trainer and pass it.
- If you create a model outside the trainer, make sure to not pass to the trainer any additional keyword arguments that are relative to `from_pretrained()` method.



## class trl.SFTTrainer

Arguments:

-  `model`: *typing.Union[transformers.modeling_utils.PreTrainedModel, torch.nn.modules.module.Module, str] = None* -> model to be train. The model can be also converted to a `PeftModel` if a `PeftConfig` object is passed to the `peft_config` argument.

- `args`: *TrainingArguments = None* -> The arguments to tweak for training. Look at `TrainingArguments` for more info.

- `data_collator`: *typing.Optional[DataCollator] = None* -> The daata collator to use for training.

- `train_dataset`: *typing.Optional[datasets.arrow_dataset.Dataset] = None* -> The dataast use for training. (raccomandation to use `trl.trainer.ConstantLengthDataset`)

- `eval_dataset`: *typing.Union[datasets.arrow_dataset.Dataset, typing.Dict[str, datasets.arrow_dataset.Dataset], NoneType] = None* -> The dataast use for evaluation. (raccomandation to use `trl.trainer.ConstantLengthDataset`)

- `tokenizer`: *typing.Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None* -> The tokenizer to use for training. **If not specified, the tokenizer associated t the model will be used.**

- `model_init`: *typing.Union[typing.Callable[[], transformers.modeling_utils.PreTrainedModel], NoneType] = None* -> The model initializer to use for training. **If None is specified, the default model initializer will be used.**

- `compute_metrics`: *typing.Union[typing.Callable[[transformers.trainer_utils.EvalPrediction], typing.Dict], NoneType] = None* -> The metrics to use for evaluation. **If no metrics are specified, the default metric (`compute_accuracy`) will be used.**

- `callbacks`: *typing.Optional[typing.List[transformers.trainer_callback.TrainerCallback]] = None* -> The callbacks to use for training.

- `optimizers`: *typing.Tuple[torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)* -> The optimizer and scheduler to use for training.

- `preprocess_logits_for_metrics`: *typing.Union[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], NoneType] = None* -> The function to use to preprocess the logits before computing the metrics.

- `peft_config`: *typing.Optional[typing.Dict] = None* -> The `PeftConfig` object to use to initialize the `PeftModel`.

- `dataset_text_field`: *typing.Optional[str] = None* -> The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a `ConstantLengthDataset` based on the `dataset_text_field` argument.

- `formatting_func`: *typing.Optional[typing.Callable] = None* -> The formatting function to be used for creating the `ConstantLengthDataset`.

- `max_seq_length`: *typing.Optional[int] = None* -> The maximum sequence length to use for the `ConsantLengthDataset` and for automatically creating the Dataset. **Defaults to 512.**

- `infinite`: *typing.Optional[bool] = False* -> Whether to use an infinite dataset or not. **Defaults to False.**

- `num_of_sequences`: *typing.Optional[int] = 1024* -> The number of sequences to use for the `CostantLengthDataset`. **Defaults to 1024.**

- `chars_per_token`: *typing.Optional[float] = 3.6* -> The number of characters per token to use for the `ConstantLengthDataset`. **Defaults to 3.6**. You can check how this is computed in the stack-llama example:
[StackLLama](https://github.com/lvwerra/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53)

- `packing`: *typing.Optional[bool] = False* -> Used only in case of `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences of the dataset.



# class transformers.TrainingArguments

Arguments:

- `output_dir` : *str* -> The output directory where the model predictions and checkpoints will be written.

- `overwrite_output_dir` : *bool = False* -> **If True**, overwrite the content of the output directory. Use this to continue if `output_dir` points to a checkpoint directory.

- `do_train` : *bool = False* -> Whether to run training or not. This argument is not directly used by **Trainer**, it's intended to be used by your training/evaluation scripts instead.

- `do_eval` : *bool = False* -> Whether to run evaluation on the validation set or not. Will be set to **True** if `evaluation_strategy` is different from **"no"**. This argument is not directly used by **Trainer**, it's intended to be used by your training/evaluation scripts instead.

- `do_predict` : *bool = False* -> Whether to run predictions on the test set or not. This argument is not directly used by **Trainer**, it's intended to be used by your training/evaluation script instead.

- `evaluation_strategy` : *typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'no'* -> The evaluation strategy to adopt during training. Possible values are:
  - **"no"** : No evaluation is done during training.
  - **"steps"** : Evaluation is done (and logged) every `eval_steps`.
  - **"epoch"** : Evaluation is done at the end of each epoch.

- `prediction_loss_only` : *bool = False* -> When performing evaluation and generating predictions, only return the loss.

- `per_device_train_batch_size` : *int = 8* -> The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.

- `per_device_eval_batch_size` : *int = 8* -> The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.

- `per_gpu_train_batch_size` : *typing.Optional[int] = None* ->
- `per_gpu_eval_batch_size` : *typing.Optional[int] = None* ->

- `gradient_accumulation_steps` : *int = 1* -> Number of updates steps to accumulate the gradients for, before performing a backward/update pass. **When useing gradient accumulation, one step is conted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step training examples.

- `eval_accumulation_steps` : *typing.Optional[int] = None* -> Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/NTU/TPU before being moved to the CPU (faster but requires more memory).

- `eval_delay` : *typing.Optional[float] = 0* -> Number of epochs or steps to wait for before the first evaluation can be performed, depending on the `evaluation_strategy`.

- `learning_rate` : *float = 5e-05* -> The initial learning rate for **AdamW** optimizer.

- `weight_decay` : *float = 0.0* -> The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in **AdamW** optimizer.

- `adam_beta1` : *float = 0.9* -> The beta1 hyperparameter for the **AdamW** optimizer.

- `adam_beta2` : *float = 0.999* -> The beta2 hyperparameter for the **AdamW** optimizer.

- `adam_epsilon` : *float = 1e-08* -> The epsilon hyperparameter for the **AdamW** optimizer.

- `max_grad_norm` : *float = 1.0* -> Maximum gradient norm (for gradient clipping).

- `num_train_epochs` : *float = 3.0* -> Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).

- `lr_scheduler_type` : *typing.Union[transformers.trainer_utils.SchedulerType, str] = 'linear'* -> The scheduler type to use. See the documentation of [SchedulerType](https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/optimizer_schedules#transformers.SchedulerType)

- `lr_scheduler_kwargs` :*typing.Optional[typing.Dict] = <factory>* -> The extra arguments for the `lr_scheduler`. See the documentation of each scheduler for possible values.

- `warmup_ratio` : *float = 0.0* -> Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.

- `warmup_steps` : *int = 0* -> Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmip_ratio`.

- `log_level` : *typing.Optional[str] = 'passive'* -> Logger log level to use on the main process. Possible choices are th log levels as string: **"debug"**, **"info"**, **"warning"**, **"error"** and **"critical"**, plus a **"passive"** level which doesn't set anything and keeps the current log level for the Transformers libray (which will be **"warning"** by default).

- `log_level_replica` : *typing.Optional[str] = 'warning'* -> Logger log level to use on replicas. Same Choices as `log_level`.

- `log_on_each_node` : *bool = True* -> In multinode distributed training, whether to log using `log_level` once per node, or only on the main node.

- `logging_dir` : *typing.Optional[str] = None* -> **TensorBoard** log directory. Will default to *output_dir/runs/CURRENT_DATETIME_HOSTNAME*.

- `logging_strategy` : *typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'steps'* -> The logging strategy to adopt during training. Possible values are:
  - **"no"** : No logging is done during training.
  - **"epoch"** : Logging is done at the end of each epoch.
  - **"steps"** : Logging is done every `logging_steps`.

- `logging_first_step` : *bool = False* -> Whether to log and evaluate the fist global_step or not.

- `logging_steps` : *float = 500* -> Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in range [0, 1). If smaller than 1, will be interpreted as ratio of total training steps.

- `logging_nan_inf_filter` : *bool = True* -> Whether to filter `nan` and `inf` losses for logging. If set to **True** the loss of every step that is `nan` or `inf is filtered and the average loss of the current logging window is taken instead.

- `save_strategy` : *typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'steps'* -> The checkpoint save strategy to adopt during training. Possible values are: 
  - **"no"** : No save is done during training.
  - **"epoch"** : Save is done at the end of each epoch.
  - **"steps"** : Save is done every `save_step`.

- `save_steps` : *float = 500* -> Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a float in range [0, 1). If smaller than 1, will be interpreted as ratio of total training steps.

- `save_total_limit` : *typing.Optional[int] = None* -> If a value  is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. When `load_best_model_at_end` is enabled, the "best" checkpoint according to `metric_for_best_model` will always be retained in addition to the most recent ones. *(For exampe, for `save_total_limit=5 and `load_best_model_at_end`, the four last checkpoints will always be retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two checkpoints are saved: the last one and the best one (if they are diffent))*.

- `save_safetensors` : *typing.Optional[bool] = True* -> Use **safetensors** saving and loading for state dicts instead of default `torch.load` and torch.save`.

- `save_on_each_node` : *bool = False* -> When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one.
This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node.

- `save_only_model` : *bool = False* -> When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state. Note that when this is true, you won't be able to resume training from checkpoint. This enables you to save storage by not strong the optimizer, scheduler & rng state. You can only load the model using `from_pretrained` with this option set to **True**.

- `no_cuda` : *bool = False* -> Whether or not to use cpu. If set to False, we will use cuda or mps device if available.

- `use_cpu` : *bool = False* -> Whether or not to use cpu. If set to False, we will use cuda or mps device if available.

<!-- - `use_mps_device` : *bool = False* -> This argument is deprecated.mps device will be used if it is available similar to cuda device. -->

- `seed` : *int = 42* -> Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the `Trainer.model_init` function to instantiate the model if it has some randomly initialized parameters.

- `data_seed` : *typing.Optional[int] = None* -> Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model seed.

- `jit_mode_eval` : *bool = False* -> Whether or not to use PyTorch jit trace for inference.

- `use_ipex` : *bool = False* -> Use intel extension for PyTorch when it is available. [IPEXinstallation](https://github.com/intel/intel-extension-for-pytorch)

- `bf16` : *bool = False* -> Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (`use_cpu`) or Ascending NPU. This is an experimental API and it may change.

- `fp16` : *bool = False* -> Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.

- `fp16_opt_level` : *str = 'O1'* -> For **fp16** training, Apex AMP optimization level selected in ["O0", "O1", "O2", and "O3"]. See details on the Apex decumentation.

- `half_precision_backend` : *str = 'auto'* ->  The backend to use for mixed precision training. Must be one of "auto", "apex", "cpu_amp". "auto" will use CPU/CUDA AMP or APEX depending on the PyTorch version detected, while the other choices will force the requested backend.

- `bf16_full_eval` : *bool = False* -> Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm metric values. This is an experimental API and it may change.

- `fp16_full_eval` : *bool = False* -> Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm metric values.

- `tf32` : *typing.Optional[bool] = None* -> Wheter to enable the TF32 mode, available in Ampere and newer GPU architectures. The default values on PyTorch's version default of `torch.beckends.cuda.matmul.allow_tf32`. For more details please refer to the [TF32](https://huggingface.co/docs/transformers/performance#tf32) documentation. This is an experimental API and it may change.

- `local_rank` : *int = -1* -> Rank of the process during distributed training.

- `ddp_backend` : *typing.Optional[str] = None* -> The backend to use for distributed training. Must bu one of **"nccl"**, **"ccl"**, **"gloo"**, **"hccl"**.

- `tpu_num_cores` : *typing.Optional[int] = None* -> When training on TPU, the number of TPU cores (automatically passed by launcher script).

- `tpu_metrics_debug` : *bool = False* ->
- `debug` : *typing.Union[str, typing.List[transformers.debug_utils.DebugOption]] = ''* ->

- `dataloader_drop_last` : *bool = False* -> Whether to drop the last incomplete batch (if the length of th ddataset is not divisible by the batch size) or not.

- `eval_steps` : *typing.Optional[float] = None* -> Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same values as `logging_steps` if not set. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.

- `dataloader_num_workers` : *int = 0* -> Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.

- `past_index` : *int = -1* -> Some models like *TransofrmerXL* or *XLNet* can make use of the past hidden states for their predictions. If this argument is set to a positive int, the Trainer will use the corresponding output (usually index 2) as the past state and feed it to the model at the next training step under the keyword arguments `num`.

- `run_name` : *typing.Optional[str] = None* -> A descriptor for the run. Typically used for **wandb** and **mlflow** logging.

- `disable_tqdm` : *typing.Optional[bool] = None* -> Whether or not to disable the tqdm progress bars and the table of metrics produced bu `notebook.Notebook TrainingTraker` in jupyter Notebooks. Will default to **True if the logging level is set to warm or lower (default)**, False otherwise.

- `remove_unused_columns` : *typing.Optional[bool] = True* -> Whether or not to automatically remove the colums unused by the model forward method. **(not implemented fo `TFTrainer` yet)**.

- `label_names` : *typing.Optional[typing.List[str]] = None* -> The list of keys in your dictionary of inputs that correspond to the labels. Will eventually default to the list of argument names accepted by the model that contain the word "label", except if the model used is one of the XxxForQuestionAnswering in which case it will also include the ["start_positions", "end_positions"] keys.

- `load_best_model_at_end` : *typing.Optional[bool] = False* -> Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved. See `save_total_limit`for more.
**When set to `True`, the parameters `save_strategy` needs to be the same as `evaluation_strategy`, and in the case it is "step", `save_steps` must be a round multiple of `eval_steps`.**

- `metric_for_best_model` : *typing.Optional[str] = None* -> Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different models. Must be the name of a metric returned by the evluation with or without the prefix `eval_`. **Will default to "loss" if unspecified and load_best_model_at_end=True` (to use the evaluation loss)**

- `greater_is_better` : *typing.Optional[bool] = None* -> Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models should have a greater metric or not. Will default to:
  - **True** if `metric_for_best_model` is set to a value that isn't "loss" or "eval_loss".
  - **False** if `metric_for_best_model` is not set, or set to "loss" or "eval_loss".

- `ignore_data_skip` : *bool = False* -> When resuming training, whether or not to skip the epochs and batches to get the data loading at the same stage as in the previous training. If set to *True*, the training will being faster (as that skipping step can take a long time) but will not yield the same results as the interrupted training would have.

- `fsdp` : *typing.Union[typing.List[transformers.trainer_utils.FSDPOption], str, NoneType] = ''* -> Use PyTorch Distributed Parallel Training (in distributed training only). A list of options along the following:
  - **"full_shard"**: shard parameters, gradients and optimizer states.
  - **"shard_grad_op"**: shard optimizer states and gradients.
  - **"hybrid_shard"**: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
  - **"hybrid_shard_zero2"**: Apply `SHARD_GRAD_OP` within a node, and replicate prameters across nodes.
  - **"offload"**: Offload parameters and gradients to CPUs (only compatible with **"full_shard"** and **"shard_grad_op"**).
  - **"auto_wrap"**: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.

- `fsdp_min_num_params` : *int = 0* ->

- `fsdp_config` : *typing.Optional[str] = None* -> Config to be used with fsdp (Pytorch Distributed Parallel Training). The values is either a logical of fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as *dict*. Complete option [here](https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py)

- `fsdp_transformer_layer_cls_to_wrap` : *typing.Optional[str] = None* ->

- `deepspeed` : *typing.Optional[str] = None* -> Use [Deepspeed](https://github.com/microsoft/deepspeed). This is an experimental feature and its API may evolve in the future. The value is either the location of DeepSpeed json config file (e.g., `ds_config.json`) or an already loaded json file as a *dict*.

- `label_smoothing_factor` : *float = 0.0* -> The label smooting factor to use. Zero means no label smoothing, otherwise the underlying onehotencoded labels are changed 0s and 1s to `label_smoothing_factor`/`num_labels` and 1 - `label_smoothing_factor` + `label_smoothing_factor`/`num_labels` respectively.

- `optim` : *typing.Union[transformers.training_args.OptimizerNames, str] = 'adamw_torch'* -> The optimizer to use: `adamw_hf`, `adamw_torch`, `adamw_torch_fused`, `adamw_apex_fused`, `adamw_anyprecision` or adafactor.

- `optim_args` : *typing.Optional[str] = None* -> Optional arguments that are supplied to **AnyPrecisionAdamW**.

- `adafactor` : *bool = False* ->

- `group_by_length` : *bool = False* -> Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.

- `length_column_name` : *typing.Optional[str] = 'length'* -> Column name for precomputed lengths. If the column exists, grouping by length will use these values rather than computing them on train startup. Ignored unless `group_by_length` is **True** and the dataset is an instance of **Dataset**.

- `report_to` : *typing.Optional[typing.List[str]] = None* -> The list of integrations to report results and logs to. Supported platoforms are **"azure_ml"**, **"clearml"**, **"codecarbon"**, **"comet_ml"**, **"dagshub"**, **"dvclive"**, **"flyte"**, **"mlflow"**, **"neptune"**, **"tensorboard"**, and **"wandb"**. Use **"all"** to report to all integrations installed, **"none"** for no intgrations.

- `ddp_find_unused_parameters` : *typing.Optional[bool] = None* -> When using distributed training, the value of the flag `find_unused_parameters` passed to `DistributedDataParallel`. Will default to **False** if gradient checkpointing is used, **True** otherwise.

- `ddp_bucket_cap_mb` : *typing.Optional[int] = None* -> When useing distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.

- `ddp_broadcast_buffers` : *typing.Optional[bool] = None* -> When using distributed training, the value of the flag `bradcast_buffers` passed to `DistributedDataParallel`. Will default to **False** if gradeint checkpointing is used, **True** otherwise.

- `dataloader_pin_memory` : *bool = True* -> Whether you want to pin memory in data loaders or not. **Will default to True.**

- `dataloader_persistent_workers` : *bool = False* -> It **True**, the dataloader will not shoutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage. **Will default to False.**

- `skip_memory_metrics` : *bool = True* -> Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows down the training and evaluation speed.

- `use_legacy_prediction_loop` : *bool = False* ->
- `push_to_hub` : *bool = False* ->

- `resume_from_checkpoint` : *typing.Optional[str] = None* -> The path to a folder with a valid checkpoint for your model. This argument is not directly used by Trainer, it's intended to be used by your training/evaluation scripts instead.

- `hub_model_id` : *typing.Optional[str] = None* ->
- `hub_strategy` : *typing.Union[transformers.trainer_utils.HubStrategy, str] = 'every_save'* ->
- `hub_token` : *typing.Optional[str] = None* ->
- `hub_private_repo` : *bool = False* ->
- `hub_always_push` : *bool = False* ->

- `gradient_checkpointing` : *bool = False* -> If **True**, use gradient checkpointing to save memory at the expense of slower backward pass.

- `gradient_checkpointing_kwargs` : *typing.Optional[dict] = None* -> Keyword arguments to be passed to the `gradient_checkpointing_enable` method.

- `include_inputs_for_metrics` : *bool = False* -> Whether or not the inputs will be passed to the `compute_metrics` function. This is intended for metrics that need inputs, predictions and references for scoring calculation in Metric class.

- `fp16_backend` : *str = 'auto'* ->
- `push_to_hub_model_id` : *typing.Optional[str] = None* ->
- `push_to_hub_organization` : *typing.Optional[str] = None* ->
- `push_to_hub_token` : *typing.Optional[str] = None* ->
- `mp_parameters` : *str = ''* ->

- `auto_find_batch_size` : *bool = False* -> Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors. **Need accelerate library**

- `full_determinism` : *bool = False* -> If **True** `enable_full_determinism()` is called instead of `set_seed()` to ensure reproducible results in distributed training. Important: this will negatively impact the performance, so only use it for debugging.

- `torchdynamo` : *typing.Optional[str] = None* -> If set the backend compiler for TorchDynamo. Possible choices are **"eager"**, **"aot_eager"**, **"introductor"**, **"nvfuser"**, **"aot_nvfuser"**, **"aot_cudagraphs"**, **"ofi"**, **"fx2trt"**, **"onnxrt"** and **"ipex"**.

- `ray_scope` : *typing.Optional[str] = 'last'* -> The scope to use when doing hyperparameter search with Ray. By default, "last" will be used. Ray will then use the last checkpoint of all trials, compare those, and select the best one. However, other options are also available. See the Ray documentation for more options.

- `ddp_timeout` : *typing.Optional[int] = 1800* -> The timeout for torch.distributed.init_process_group calls, used to avoid GPU socket timeouts when performing slow operations in distributed runnings. Please refer the [PyTorch documentation] (https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more information.

- `torch_compile` : *bool = False* ->  Whether or not to compile the model using PyTorch 2.0 torch.compile.
This will use the best defaults for the torch.compile API. You can customize the defaults with the argument torch_compile_backend and torch_compile_mode but we don’t guarantee any of them will work as the support is progressively rolled in in PyTorch.
This flag and the whole compile API is experimental and subject to change in future releases.

- `torch_compile_backend` : *typing.Optional[str] = None* -> The backend to use in torch.compile. If set to any value, torch_compile will be set to True.
Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.
This flag is experimental and subject to change in future releases.

- `torch_compile_mode` : *typing.Optional[str] = None* -> The mode to use in torch.compile. If set to any value, torch_compile will be set to True.
Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.
This flag is experimental and subject to change in future releases.

- `dispatch_batches` : *typing.Optional[bool] = None* ->

- `split_batches` : *typing.Optional[bool] = False* -> Whether or not the accelerator should split the batches yielded by the dataloaders across the devices during distributed training. If set t **True**, the actual batch size used will be the same on any kind of distributed processes, but it must be a round multiple of the number of processes you are using (such as GPUs).

- `include_tokens_per_second` : *typing.Optional[bool] = False* -> Whether or not to compute the number of tokens per second per device for training speed metrics. This will iterate over the entire training dataloader once beforehand, and will slow down the entire process.

- `include_num_input_tokens_seen` : *typing.Optional[bool] = False* -> Whether or not to track the number of input tokens seen throughout training.
May be slower in distributed training as gather operations must be called.

- `neftune_noise_alpha` : *float = None* -> **If not None, this will activate NEFTune noise embeddings. This can drastically improve model performance for instruction fine-tuning. Check out the [original paper](https://arxiv.org/abs/2310.05914) and the [original code](https://github.com/neelsjain/NEFTune). Support transformers PreTrainedModel and also PeftModel from peft.**