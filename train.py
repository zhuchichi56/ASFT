import os
import copy
import logging
import json
import numpy as np
import fire
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import os
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

IGNORE_INDEX = -100

def load_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b")

@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data.json")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=3.0)
    learning_rate: float = field(default=2e-5)

class EnhancedTrainer(Trainer):
    def __init__(self, mode="sft", kl_weight=0.1, clip_min=0.1, clip_max=2.0, original_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.kl_weight = kl_weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.original_model = original_model
        if original_model is not None:
            self.original_model.eval()
        print(f"Training mode: {mode}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            valid_mask = shift_labels != IGNORE_INDEX
            
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)
                
                if self.mode == "sft":
                    weighted_losses = token_losses
                    
                elif self.mode == "dft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    weighted_losses = token_losses * weights
                    
                elif self.mode == "sft+kl":
                    if self.original_model is not None:
                        with torch.no_grad():
                            orig_outputs = self.original_model(**inputs)
                            orig_logits = orig_outputs.get("logits")[..., :-1, :].contiguous()
                            # Ensure sequence lengths match
                            orig_logits = orig_logits.view(-1, orig_logits.size(-1))[:shift_logits.size(0)]
                        if orig_logits.size(0) == shift_logits.size(0):
                            kl_div = F.kl_div(F.log_softmax(shift_logits, dim=-1), F.softmax(orig_logits, dim=-1), reduction='none').sum(dim=-1)
                            weighted_losses = token_losses + self.kl_weight * kl_div
                        else:
                            weighted_losses = token_losses
                    else:
                        weighted_losses = token_losses
                                            
                elif self.mode == "asft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    dft_losses = token_losses * weights
                    if self.original_model is not None:
                        with torch.no_grad():
                            orig_outputs = self.original_model(**inputs)
                            orig_logits = orig_outputs.get("logits")[..., :-1, :].contiguous()
                            orig_logits = orig_logits.view(-1, orig_logits.size(-1))[:shift_logits.size(0)]
                        if orig_logits.size(0) == shift_logits.size(0):
                            kl_div = F.kl_div(F.log_softmax(shift_logits, dim=-1), F.softmax(orig_logits, dim=-1), reduction='none').sum(dim=-1)
                            weighted_losses = dft_losses + self.kl_weight * kl_div
                        else:
                            weighted_losses = dft_losses
                    else:
                        weighted_losses = dft_losses
                                                    
                loss = weighted_losses[valid_mask].mean()
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_jsonl(data_path)
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def show_first_example(data_path: str, tokenizer: transformers.PreTrainedTokenizer):
    """Show first training example with tokenization details"""
    print("\n" + "="*50)
    print("FIRST TRAINING EXAMPLE")
    print("="*50)
    
    # Load first example
    data = load_jsonl(data_path)
    if not data:
        print("No data found")
        return
    
    example = data[0]
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    print(f"Instruction: {instruction}")
    print(f"Response: {response}")
    
    # Format with prompt
    prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    full_text = prompt + response + tokenizer.eos_token
    
    print(f"\nFull prompt:\n{prompt}")
    print(f"Target text: {response}{tokenizer.eos_token}")
    
    # Tokenize
    tokenized = tokenizer(full_text, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    input_ids = tokenized.input_ids[0]
    
    # Calculate instruction length for masking
    instruction_tokenized = tokenizer(prompt, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    instruction_len = instruction_tokenized.input_ids[0].shape[0]
    
    print(f"\nTokenization:")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Instruction tokens: {instruction_len}")
    print(f"Response tokens: {len(input_ids) - instruction_len}")
    
    # Show loss computation part
    labels = input_ids.clone()
    labels[:instruction_len] = IGNORE_INDEX
    
    print(f"\nLoss computation tokens (response part):")
    loss_tokens = input_ids[instruction_len:]
    decoded_loss_part = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    print(f"Tokens for loss: {decoded_loss_part}")
    print(f"Token IDs: {loss_tokens.tolist()}")
    print("="*50 + "\n")


def train(
    model_name_or_path: str = "models/Llama-2-7b",
    data_path: str = "data/train_medmcqa_alpaca_10k.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 4,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    mode: str = "sft",  # sft, dft, sft+kl, asft
    kl_weight: float = 0.1,
    clip_min: float = 0.1,
    clip_max: float = 2.0,
    output_dir: str = None,
    **kwargs
):
    """Enhanced training with multiple DFT variants"""
    
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)
    
    if output_dir is None:
        output_dir: str = f"./output/{mode}/{os.path.basename(model_name_or_path)}"
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // (per_device_train_batch_size * world_size)

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        # save_steps=4,
        save_strategy="epoch",
        save_total_limit=50,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # logging_steps=1,
        **kwargs
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        
    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })

    # Show first example
    show_first_example(data_args.data_path, tokenizer)

    # Load original model for KL modes
    original_model = None
    if "kl" in mode or mode == "asft":
        print("Loading original model for KL divergence...")
        original_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        original_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if original_tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=original_tokenizer,
                model=original_model,
            )

        for param in original_model.parameters():
            param.requires_grad = False
        original_model.eval()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args) 
    trainer = EnhancedTrainer(
        mode=mode,
        kl_weight=kl_weight,
        clip_min=clip_min,
        clip_max=clip_max,
        original_model=original_model,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
    