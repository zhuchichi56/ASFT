#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

from vllm import LLM, SamplingParams
from verl.utils.med_mcq import extract_answer_letter_fallback, render_mcq_prompt


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_question_medqa(item: Dict[str, Any]) -> str:
    return render_mcq_prompt(item["question"], item["options"])


def format_question_mmlu(item: Dict[str, Any]) -> str:
    return render_mcq_prompt(item["question"], item["choices"])


def format_question_medmcqa(item: Dict[str, Any]) -> str:
    return render_mcq_prompt(item["question"], [item["opa"], item["opb"], item["opc"], item["opd"]])


def get_answer_from_logprobs(output, llm) -> str:
    text_prediction = extract_answer_letter_fallback(output.outputs[0].text)

    if not hasattr(output.outputs[0], "logprobs") or output.outputs[0].logprobs is None or len(output.outputs[0].logprobs) == 0:
        return text_prediction

    logprobs = output.outputs[0].logprobs[0]
    option_probs = {}
    tokenizer = llm.get_tokenizer()

    for token_id, logprob_data in logprobs.items():
        token = tokenizer.decode([token_id]).strip().upper()
        if token in ["A", "B", "C", "D"]:
            option_probs[token] = logprob_data.logprob

    if option_probs:
        return max(option_probs, key=option_probs.get)
    return text_prediction


def get_correct_answer(item: Dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "medqa":
        return item["answer_idx"]
    if dataset_type == "mmlu":
        return chr(65 + item["answer"])
    if dataset_type == "medmcqa":
        return chr(65 + item["cop"]) if item["cop"] != -1 else "A"
    return "A"


def test_dataset(
    llm: LLM,
    data: List[Dict[str, Any]],
    dataset_type: str,
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    if dataset_type == "medqa":
        prompts = [format_question_medqa(item) for item in data]
    elif dataset_type == "mmlu":
        prompts = [format_question_mmlu(item) for item in data]
    else:
        prompts = [format_question_medmcqa(item) for item in data]

    outputs = llm.generate(prompts, sampling_params)
    correct = 0
    total = len(data)
    predictions: List[Dict[str, str]] = []

    for i, output in enumerate(outputs):
        predicted = get_answer_from_logprobs(output, llm)
        correct_answer = get_correct_answer(data[i], dataset_type)
        if predicted == correct_answer:
            correct += 1
        predictions.append(
            {
                "question": data[i]["question"],
                "predicted": predicted,
                "correct": correct_answer,
            }
        )
    return {"accuracy": correct / total, "predictions": predictions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset", choices=["medqa", "mmlu", "medmcqa", "all"], default="all")
    args = parser.parse_args()

    datasets = {
        "medqa": os.path.join(args.test_data_dir, "medqa_test.jsonl"),
        "mmlu": os.path.join(args.test_data_dir, "mmlu_medical_test.jsonl"),
        "medmcqa": os.path.join(args.test_data_dir, "medmcqa_test.jsonl"),
    }

    selected_datasets = datasets if args.dataset == "all" else {args.dataset: datasets[args.dataset]}

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=10)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    results = {}
    try:
        model_results = {}
        for dataset_name, dataset_path in selected_datasets.items():
            data = load_jsonl(dataset_path)
            if args.max_samples is not None:
                data = data[: args.max_samples]
            dataset_result = test_dataset(llm, data, dataset_name, sampling_params)
            accuracy = dataset_result["accuracy"]
            model_results[dataset_name] = {
                "accuracy": accuracy,
                "total_samples": len(data),
                "predictions": dataset_result["predictions"],
            }
            print(f"{dataset_name}: {accuracy:.4f} ({int(accuracy * len(data))}/{len(data)})")
        results[args.model] = model_results
    finally:
        del llm

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
