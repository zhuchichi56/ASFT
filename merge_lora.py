#!/usr/bin/env python
# merge_lora.py - Merge LoRA adapters into the base model

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from train_lora import smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN


def merge_lora(
    base_model_path: str,
    lora_model_path: str,
    save_path: str,
    dtype=torch.float16,
):
    """
    Merge LoRA adapter into the base model and save the merged model.

    Args:
        base_model_path (str): Path to the pretrained base model.
        lora_model_path (str): Path to the trained LoRA model (PEFT).
        save_path (str): Path to save the merged model.
        dtype: torch dtype for saving (default: float16)
    """

    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=base_model,
        )

    print(f"Loading LoRA adapter from: {lora_model_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=dtype)

    print("Merging LoRA adapter into base model...")
    merged_model = lora_model.merge_and_unload()
    
    # Save merged model
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving merged model to: {save_path}")
    merged_model.save_pretrained(save_path)

    # Save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)

    print("Merge complete!")

if __name__ == "__main__":
    import fire

    fire.Fire({
        "merge_lora": merge_lora
    })
