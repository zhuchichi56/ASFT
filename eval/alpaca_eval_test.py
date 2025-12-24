import os
import subprocess
from loguru import logger
import datasets
import json
from inference_utils import parallel_inference
# from mt_eval import curl_google, check_openai_api_key
import torch
import os

import sys
sys.path.append("./")
sys.path.append("/..")
sys.path.append("/..")


import os

# 直接使用 os.environ 设置环境变量
os.environ["OPENAI_API_BASE"] = "s"
os.environ["OPENAI_API_KEY"] = ""

def curl_google():
    response = subprocess.run(
        ["curl", "-I", "--max-time", "5", "https://www.google.com"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if response.returncode == 0:
        print("Curl Google Success")
    else:
        raise Exception(f"Curl Error: Connection to Google failed")

def check_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API key is set.")
    else:
        raise Exception("OpenAI API key is not set.")
    


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
def resolve_path(relative_path):
    return os.path.join(ROOT_DIR, relative_path)

BASE_PATH: str = resolve_path("alpacaeval")
print(BASE_PATH)
RESULT_PATH: str = os.path.join(BASE_PATH, "results")


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

STRATEGY_KEYWORDS = [
    "step_cosine_entropy_gate",
    "step_linear_prob_inverse",
    "step_exp_loss_boost",
    "warmup_cosine_floor",
    "adaptive_log_decay",
    "gated_entropy_step",
    "long_sft_step",
    "delayed_cosine_high",
    "mid_point_fast",
    "boosted_cold_start",
    "fixed_05",
    "epoch_step",
    "half_epoch_decay",
    "linear_full",
    "celoss",
    "coldstart",
    "cold_start",
    "entropy",
    "entropydecay",
    "sft",
]


def discover_models(trained_root: str):
    discovered = {}
    if not os.path.isdir(trained_root):
        return discovered
    for entry in os.listdir(trained_root):
        full_path = os.path.join(trained_root, entry)
        if not os.path.isdir(full_path):
            continue
        for strategy in STRATEGY_KEYWORDS:
            if f"-{strategy}" in entry:
                discovered[full_path] = strategy
                break
    return discovered


def alpaca_eval_model(
    reference_model: str = "gpt4_1106_preview",
    model_name_or_path: str = None,
    custom_name: str = None,  # Added custom_name parameter
    result_pth: str = RESULT_PATH,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    top_p: float = 1,
    top_k: int = None,
    template_type: str = "default",
    gpus=None,
    # only_inference: bool = True
):
    only_inference = torch.cuda.is_available()
    custom_name = custom_name or model_name_or_path.split("/")[-1]
    
    result_pth = os.path.join(result_pth, custom_name)
    result_file_path = os.path.join(result_pth, "model_outputs.json")
    eval_dataset_path = os.path.join(BASE_PATH, "alpaca_eval.json")
    if not os.path.isfile(eval_dataset_path):
        raise FileNotFoundError(f"{eval_dataset_path} not found, please ensure the AlpacaEval data exists locally.")
    eval_set = datasets.load_dataset(
        "json",
        data_files={"eval": eval_dataset_path},
        cache_dir=BASE_PATH,
    )["eval"]

    # if has_gpu or use_gpt:
    if only_inference:
        # Ensure evaluation output
        if not os.path.exists(result_file_path):
            if not os.path.exists(model_name_or_path):
                print(f"Model {model_name_or_path} does not exist.")
                raise ValueError(f"Model {model_name_or_path} does not exist.")
            eval_prompts = [example["instruction"] for example in eval_set]
            eval_responses = parallel_inference(
                prompt_list=eval_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                template_type=template_type,
                model_path=model_name_or_path,
                gpus=gpus,
            )
            result_file = [
                {
                    "dataset": example["dataset"],
                    "instruction": example["instruction"],
                    "output": response,
                    "generator": custom_name,
                }
                for example, response in zip(eval_set, eval_responses)
            ]
            os.makedirs(result_pth, exist_ok=True)
            save_json(result_file, result_file_path)
        else:
            print(f"{result_pth} already exists, skipping model outputs generation.")
        return

    if not os.path.exists(result_file_path):
        raise FileNotFoundError(f"{result_file_path} not found, please run inference on a GPU machine first.")


    reference_paths = {
        "llama3-8b-instruct": "alpacaeval/results/Meta-Llama-3-8B-Instruct_origin",
        "llama3-70b-instruct": "alpacaeval/results/Meta-Llama-3-70B-Instruct",
        "llama2-7b-chat": "alpacaeval/results/llama-2-7b-chat-hf",
        "llama2-13b-chat": "alpacaeval/results/llama-2-13b-chat-hf",
        "llama2-70b-chat": "alpacaeval/results/llama-2-70b-chat-hf",
        "mistral-7b-instruct-v0.2": "alpacaeval/results/Mistral-7B-Instruct-v0.2",
        "mistral-7b-instruct-v0.3": "alpacaeval/results/Mistral-7B-Instruct-v0.3",
        "gpt4_1106_preview": "alpacaeval/results/gpt4_1106_preview",
    }
    reference_paths = {k: resolve_path(v) for k, v in reference_paths.items()}
    
    reference_outputs = reference_paths.get(reference_model.lower(), reference_paths["gpt4_1106_preview"])

    # Pre-run checks
    curl_google()
    check_openai_api_key()

    # Annotator setup
    # annotator_model = "weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
    # annotator_model = "gpt4_1106_preview"
    annotator_model = "weighted_alpaca_eval_gpt-4o-2024-08-06"
    logger.info(f"The Annotator Model is {annotator_model}, The reference model is {reference_model}, The reference_outputs is {reference_outputs}")

    command = (
        f"PYTHONPATH=$(pwd)/alpacaeval && python -m "
        "alpacaeval.src.alpaca_eval.main "
        f"--model_outputs {result_file_path} "
        f"--reference_outputs {os.path.join(reference_outputs, 'model_outputs.json')} "
        f"--output_path {os.path.join(result_pth, f'alpaca_eval_output_{annotator_model}_vs_{reference_model}.json')} "
        f"--annotators_config {annotator_model} "
        f"--caching_path {result_pth}/annotations_cache_annotator_{annotator_model}_vs_{reference_model}.json",
        f"--max_instances 32"
    )
    logger.info(f"Running command: {command}")
    subprocess.run(command, shell=True)



if __name__ == "__main__":

    
    model_name_or_path_list =[
        "",
        ""
    ]
    for model_name_or_path in model_name_or_path_list:
        try:
            alpaca_eval_model(
                model_name_or_path=model_name_or_path,
                custom_name=custom_names.get(model_name_or_path, None),
                reference_model="gpt4_1106_preview",
                temperature=0,
                template_type="alpaca",
                gpus ="0,1,2,3"
            )
        except Exception as e:
            print(f"Error evaluating model {model_name_or_path}: {str(e)}")
