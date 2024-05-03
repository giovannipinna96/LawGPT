from typing import Optional, Any

import torch

from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)

from peft import PeftModel
import pandas as pd
import json
import time

ALPACA_TEMPLATE = "[INST] <<SYS>>\nSei un modello esperto di leggi, sentenze e ordinanze italiane. Di seguito è riportata una sentenza o una legge italiana, il tuo compito è quello di riassumerla. Scrivi una risposta che soddisfi adeguatamente questa richiesta.\n<</SYS>>\n\n{input} [/INST] "


def load_adapted_hf_generation_pipeline(
    base_model_name,
    lora_model_name,
    temperature: float = 0.6,
    top_p: float = 1.0,
    max_tokens: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    load_in_8bit: bool = True,
    generation_kwargs: Optional[dict] = None,
):
    # """
    # Load a huggingface model & adapt with PEFT.
    # Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    # """

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if load_in_8bit and not is_bitsandbytes_available():
        raise ValueError("Install `bitsandbytes`")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"

    # if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=load_in_8bit,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_model_name,
        # torch_dtype=torch.float16,
    )
    # elif device == "mps":
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model_name,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_model_name,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model_name, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_model_name,
    #         device_map={"": device},
    #     )

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_in_8bit:
    #     model.half()  # seems to fix bugs for some users.

    # model.eval()

    # generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        # do_sample=True,
        temperature=temperature,
        # max_new_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=1.2,
        # **generation_kwargs,
    )
    # pipe = pipeline(
    #     task,
    #     model=model,
    #     tokenizer=tokenizer,
    #     batch_size=16, # TODO: make a parameter
    #     generation_config=config,
    #     framework="pt",
    # )

    return model, tokenizer, config


if __name__ == "__main__":
    # model, tokenizer, config = load_adapted_hf_generation_pipeline(
    base_model_name = "swap-uniba/LLaMAntino-2-7b-hf-ITA"
    lora_model_name = "./llamantino7summarization"
    # )
    temperature: float = 0.6
    top_p: float = 0.95
    load_in_8bit: bool = True

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # task = "text-generation"

    # if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=load_in_8bit,
        # torch_dtype=torch.float16,
        # device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_model_name,
        # torch_dtype=torch.float16,
    )

    config = GenerationConfig(
        # do_sample=True,
        temperature=temperature,
        # max_new_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=1.2,
    )

    print("Load dataset")
    df = pd.read_json("lawgpt/data/processed/summarization/test_data.json")
    df = df.head(200)
    df_dict = df.to_dict(orient="records")

    print("strat generating")
    i = 0
    model.eval()
    response = []
    for i in range(len(df_dict)):
        prompt = ALPACA_TEMPLATE.format(input=df_dict[i]["reference"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                # max_length=512
            )
        print("generating ...")
        for seq in generation_output.sequences:
            output = tokenizer.decode(seq)
            # print("Risposta:", output) #.split("[/INST]")[1].strip())
        # response.append({'request': row['reference'],
        #                  'gold': row['summary'],
        #                  'generated_total' : output,
        #                  'generated_response': output.split("[/INST]")[1].strip()})
        response.append(output)
        # file_path = 'my_dict2.json'

        # # Save the dictionary as a JSON file
        # with open(file_path, 'a') as json_file:
        #     json.dump({'index': i,
        #             'request': df_dict[i]['reference'],
        #                 'gold': df_dict[i]['summary'],
        #                 'generated_total' : output,
        #                 'generated_response': output.split("[/INST]")[1].strip()}, json_file)
        #     json_file.write('\n')
        print("=" * 30)
        print(i)
        print("=" * 30)
        i = i + 1
    # del generation_output
    # del input_ids
    # del inputs
    # torch.cuda.empty_cache()
    # time.sleep(5)


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline
# from peft import PeftModel, PeftConfig
# import pandas as pd

# print("start")
# model_id = "./merge_llamantino7"
# tokenizer = AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-2-7b-hf-ITA")
# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
# print("modello caricato")
# # model = PeftModel.from_pretrained(model, "./llamantino7summarization")
# print("peft caricato")
# pipe = pipeline("text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=200,
#                 device=0
# )
# print("pipeline fatta")
# df = pd.read_json("lawgpt/data/processed/summarization/test_data.json")
# df = df.head(200)
# df_dict = df.to_dict(orient='records')

# print("strat generating")
# i = 0
# response = []
# for i in range(len(df_dict)):
#     print("=" * 30)
#     print(i)
#     print("=" * 30)
#     prompt = df_dict[i]['reference']
#     print("inizio a generare")
#     try:
#         r = pipe(prompt)[0]['generated_text']
#         print(r)
#         response.append(r)
#     except:
#         pass
