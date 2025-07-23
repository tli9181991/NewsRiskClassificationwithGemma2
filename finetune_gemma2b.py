import os
import gc
import torch
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainingArguments
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

def jsonstr2dict(json_str):
    json_str = json_str.replace("```", "")
    json_str = json_str.replace("\n", "")
    json_str = json_str.replace("json", "")

    return eval(json_str)

def dict2ans(rank_dict):
    item_list = list(rank_dict.items())
    ans_str = ""
    for risk, rank in item_list:
        ans_str += f"{risk}: {rank}\n"

    return ans_str

def jsonstr2ans(json_str):
    ans_dict = jsonstr2dict(json_str)
    return dict2ans(ans_dict)

def parse_opt():
    parser = argparse.ArgumentParser(description="training arguments")
    parser.add_argument("--train_path", type=str, default="./data/train.csv", help="Path to train csv file")
    parser.add_argument("--test_path", type=str, default="./data/test.csv", help="Path to test csv file")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--train_prompt_style", type=str, default="./train_prompt_style.txt", help="train prompt style txt file")
    parser.add_argument("--influence_prompt_style", type=str, default="./influence_prompt_style.txt", help="influence prompt style txt file")
    parser.add_argument("--output", type=str, default="Output", help="output folder")

    return parser.parse_args()

def load_prompl_stype(path):
    prompt_style = """"""
    with open(path, 'r') as f:
        prompt_style = f.read()

    if prompt_style[0] != '\n':
        prompt_style = '\n' + prompt_style

    if prompt_style[-1] != '\n':
        prompt_style = prompt_style + '\n'

    return prompt_style

def fine_tune_gemma2b(params):
    df_train = pd.read_csv(params.train_path)
    df_test = pd.read_csv(params.test_path)

    train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
    test_dataset = Dataset.from_pandas(df_test, preserve_index=False)

    model_id = "google/gemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                quantization_config=bnb_config,
                                                device_map="auto",
                                                token=os.environ['HF_TOKEN'])
    
    train_prompt_style = load_prompl_stype(params.train_prompt_style)
    inference_prompt_style = load_prompl_stype(params.inference_prompt_style)

    def formatting_prompts_func(examples):
        tickers = examples["ticker"]
        headlines = examples["headline"]
        descriptions = examples["description"]
        responses = examples["answer"]
        texts = []
        for ticker, headline, description, response in zip(tickers, headlines, descriptions, responses):
            
            # Append the EOS token to the response if it's not already there
            response = "<answer>\n\n" + response + "\n</answer>\n"
            response += tokenizer.eos_token
                
            text = train_prompt_style.format(ticker, headline, description, response)
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    test_dataset = test_dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    response_template = "<answer>"
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template
    )

    training_arguments = TrainingArguments(
        output_dir="NewsRiskRanking",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=params.epochs,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=5e-5,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="none",
        label_names=["labels"],  # add this
        save_strategy="steps",            # or "epoch"
        save_steps=100,                   # save every 500 steps
    )

    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    trainer.train()

    trainer.save_model(params.output)

if __name__ == "__main__":
    params = parse_opt()
    fine_tune_gemma2b(params)