from typing import List, Dict, Optional, Sequence, Union
import os
import numpy as np
from vllm import LLM, SamplingParams
from scipy.special import softmax
from loguru import logger 
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from transformers import AutoTokenizer
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_gpus(gpu_spec: Optional[Union[str, Sequence[str]]] = None) -> List[str]:
    """
    Resolve the GPU list to use for inference. Priority:
    1) Explicit gpu_spec (comma-separated string or sequence)
    2) CUDA_VISIBLE_DEVICES environment variable
    3) torch.cuda.device_count()
    """
    if isinstance(gpu_spec, str):
        candidates = gpu_spec.split(",")
    elif gpu_spec is not None:
        candidates = gpu_spec
    else:
        env_val = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_val:
            candidates = env_val.split(",")
        elif torch.cuda.is_available():
            candidates = [str(i) for i in range(torch.cuda.device_count())]
        else:
            candidates = []

    gpus = [str(g).strip() for g in candidates if str(g).strip()]
    if not gpus:
        raise RuntimeError("No GPUs available for inference; set CUDA_VISIBLE_DEVICES or pass gpus explicitly.")
    return gpus

# config_table = {
#     "llama2": {
#         "max_model_len": 2048,
#         "id2score": {29900: "0", 29896: "1"}
#     },
#     "llama3": {
#         "max_model_len": 8192,
#         "id2score": {15: "0", 16: "1"}
#     },
#     "mistral": {
#         "max_model_len": 2000,
#         "id2score": {28734: "0", 28740: "1"}
#     },
# }

# def get_model_config(model_path):
#     for key in config_table:
#         if key in model_path.lower():
#             logger.info(f"Using config for {key}")
#             return config_table[key]
#     return config_table["mistral"]

def vllm_inference(model_path: str, input_data: List[str], gpu_id: str, max_tokens: int = 256, temperature: float = 0 , top_p: float = 0.9, skip_special_tokens: bool = True):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"Process running on GPU {gpu_id}")
    
    # config = get_model_config(model_path)
    # llm = LLM(model=model_path, tokenizer_mode="auto", trust_remote_code=True, max_model_len=config["max_model_len"], gpu_memory_utilization=0.60)
    llm = LLM(model=model_path, tokenizer_mode="auto", trust_remote_code=True, gpu_memory_utilization=0.40)
    
    if "llama3" in model_path:
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens,
                                         stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")])
    else:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens)
    
    
    outputs = llm.generate(input_data, sampling_params)
    return [output.outputs[0].text for output in outputs]



def get_template(prompt, template_type="default", tokenizer=None):
    # logger.info(f"Using template type: {template_type}")
    if template_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    else:
        # messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def parallel_inference(prompt_list: List[str], 
                       model_path: str, 
                       max_tokens: int = 256, 
                       temperature: float = 0 , 
                       top_p: float = 0.9, 
                       top_k: int = None,
                       template_type: str = "alpaca",
                       skip_special_tokens: bool = True,
                       gpus: Optional[Union[str, Sequence[str]]] = None):

    gpus = resolve_gpus(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    logger.info(f"Using GPUs for inference: {gpus}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_list = [get_template(prompt, template_type=template_type, tokenizer=tokenizer) for prompt in prompt_list]
    if prompt_list:
        logger.info(f"The example prompt:\n{prompt_list[0]}") 

    chunks = [list(chunk) for chunk in np.array_split(prompt_list, len(gpus)) if len(chunk) > 0]
    active_gpus = gpus[:len(chunks)]
    if not active_gpus:
        raise RuntimeError("No work chunks created for inference; check prompts or GPU configuration.")
    
    results = []
    with ProcessPoolExecutor(max_workers=len(active_gpus), mp_context=get_context("spawn")) as executor:
        futures = executor.map(
            vllm_inference,
            [model_path] * len(active_gpus),
            chunks,
            active_gpus, 
            [max_tokens] * len(active_gpus),
            [temperature] * len(active_gpus),
            [top_p] * len(active_gpus),
            [skip_special_tokens] * len(active_gpus),
        )
        for result in futures:
            results.extend(result)
    
    return results

if __name__ == "__main__":
    model_path = "/volume/pt-train/models/Llama-3.1-8B-Instruct"
    prompt_list = ["Analyze the word choice, phrasing, punctuation, and capitalization in the given email. How may the writer of this email sound to the reader? These tones include Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, Encouraging, Egocentric, Joyful, Optimistic, and Excited.\n\nHi Jen, \nI hope you're well. Can we catch up today? I'd appreciate your input on my presentation for tomorrow's meeting. I'd especially love it if you could double-check the sales numbers with me. There's a coffee in it for you!"]
    result = parallel_inference(prompt_list, model_path, max_tokens=256, temperature=0, top_p=0.95, skip_special_tokens=True)
    print(result[0])
    
    
    
    
    
    
    
