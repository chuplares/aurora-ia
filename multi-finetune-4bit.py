import torch.distributed as dist
import argparse
import os
import sys
import torch
import pickle
import random
import json
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from datasets import load_dataset

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-o',"--output_folder", type=str, default="./aurora-out", help="Set output directory. Default: ./aurora-out")
parser.add_argument('-z',"--batch_size", type=str, default=32, help="Set batch size. Default: 32")
parser.add_argument('-b',"--micro_batch_size", type=int, default=2, help="Set micro batch size. Default: 2")
parser.add_argument('-r',"--learning_rate", type=float, default=0.0002, help="Set learning rate. Default: 0.0002")
parser.add_argument('-d',"--datasets", type=str, required=True, help="Set the name of the dataset. Default: marx")
parser.add_argument('-s',"--deepspeed", type=str, help="Set deepspeed config file")
parser.add_argument('-l',"--local_rank", type=int, default=0, help="Set local rank, if using the script for distributed training. Default: 0")
parser.add_argument('-m',"--model_path", type=str, required=True, help="Set the path for the llama/openllama model.")
parser.add_argument("--lora_r", type=int, default=8, help="Set lora rank. Default: 8")
parser.add_argument("--lora_alpha", type=int, default=32, help="Set lora alpha. Default: 32")
parser.add_argument("--max_len", type=int, default=2048, help="Set max length. Default: 2048 ")
parser.add_argument('-e',"--epoch", type=int, default=1, help="Set number of epochs. Default: 1")
args = parser.parse_args()

# Parameters
MICRO_BATCH_SIZE = args.micro_batch_size
size = args.size
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epoch
CUTOFF_LEN = args.max_len
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
]
DATA_PATH = "data/tmp.json"
OUTPUT_DIR = "checkpoints/{}".format(size)

if not os.path.exists("data"):
    os.makedirs("data")
# Load data
data = []
for x in args.datasets.split(","):
    data += json.load(open("data/{}_chat_data.json".format(x)))
random.shuffle(data)
json.dump(data, open(DATA_PATH, "w"))
data = load_dataset("json", data_files=DATA_PATH)

# Set up environment variables for distributed training
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(local_rank)


# Load Model
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_4bit=True,
    device_map=device_map,
    quantization_config=bnb_config,
)
total_params, params = 0, 0

tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
config.save_pretrained(OUTPUT_DIR)

model = get_peft_model(model, config)
tokenizer.pad_token_id = 0

for n, p in model.model.named_parameters():
    if any([x in n for x in ["lora"]]):
        total_params += p.numel()
    params += p.numel()

print(
    "Total number of parameters: {}M, rate: {}%".format(
        total_params // 1000 / 1000, round(total_params / params * 100, 2)
    )
)

# Data Preprocess
def generate_prompt(data_point):
    return data_point["input"]


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        logging_steps=1,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=50,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(OUTPUT_DIR)


