#!/usr/bin/env python3
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import ray
from vllm import LLM, SamplingParams
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_question_medqa(item: Dict[str, Any]) -> str:
    """Format MedQA question"""
    question = item['question']
    options = item['options']
    formatted = f"Question: {question}\n\nOptions:\n"
    for key, value in options.items():
        formatted += f"{key}. {value}\n"
    formatted += "\nAnswer:"
    return formatted


def format_question_mmlu(item: Dict[str, Any]) -> str:
    """Format MMLU Medical question"""
    question = item['question']
    choices = item['choices']
    formatted = f"Question: {question}\n\nOptions:\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(65+i)}. {choice}\n"
    formatted += "\nAnswer:"
    return formatted


def format_question_medmcqa(item: Dict[str, Any]) -> str:
    """Format MedMCQA question"""
    question = item['question']
    options = [item['opa'], item['opb'], item['opc'], item['opd']]
    formatted = f"Question: {question}\n\nOptions:\n"
    for i, option in enumerate(options):
        formatted += f"{chr(65+i)}. {option}\n"
    formatted += "\nAnswer:"
    return formatted


def get_answer_from_logprobs(output, llm) -> str:
    """Extract answer based on A/B/C/D token probabilities"""
    if not hasattr(output.outputs[0], 'logprobs') or output.outputs[0].logprobs is None or len(output.outputs[0].logprobs) == 0:
        return extract_answer_letter_fallback(output.outputs[0].text)
    
    logprobs = output.outputs[0].logprobs[0]  # First token logprobs
    option_probs = {}
    
    # Get tokenizer from LLM to decode token IDs
    tokenizer = llm.get_tokenizer()
    
    for token_id, logprob_data in logprobs.items():
        token = tokenizer.decode([token_id]).strip().upper()
        if token in ['A', 'B', 'C', 'D']:
            option_probs[token] = logprob_data.logprob
    
    if option_probs:
        return max(option_probs, key=option_probs.get)
    return extract_answer_letter_fallback(output.outputs[0].text)


def extract_answer_letter_fallback(response: str) -> str:
    """Fallback method to extract answer letter from response text"""
    response = response.strip().upper()
    for char in ['A', 'B', 'C', 'D']:
        if char in response:
            return char
    return 'A'


def get_correct_answer(item: Dict[str, Any], dataset_type: str) -> str:
    """Get correct answer for each dataset type"""
    if dataset_type == 'medqa':
        return item['answer_idx']
    elif dataset_type == 'mmlu':
        return chr(65 + item['answer'])
    elif dataset_type == 'medmcqa':
        return chr(65 + item['cop']) if item['cop'] != -1 else 'A'
    return 'A'


def test_dataset(llm: LLM, data: List[Dict[str, Any]], dataset_type: str, sampling_params: SamplingParams) -> float:
    """Test a dataset and return accuracy"""
    if dataset_type == 'medqa':
        prompts = [format_question_medqa(item) for item in data]
    elif dataset_type == 'mmlu':
        prompts = [format_question_mmlu(item) for item in data]
    elif dataset_type == 'medmcqa':
        prompts = [format_question_medmcqa(item) for item in data]
        
    outputs = llm.generate(prompts, sampling_params)   
    correct = 0
    total = len(data)
    
    for i, output in enumerate(outputs):
        predicted = get_answer_from_logprobs(output, llm)
        correct_answer = get_correct_answer(data[i], dataset_type)
        if predicted == correct_answer:
            correct += 1
    return correct / total




def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # List of models to test
        models = []
        
        # Dataset paths
        test_data_dir = "test_data"
        datasets = {
            'medqa': os.path.join(test_data_dir, 'medqa_test.jsonl'),
            'mmlu': os.path.join(test_data_dir, 'mmlu_medical_test.jsonl'),
            'medmcqa': os.path.join(test_data_dir, 'medmcqa_test.jsonl')
        }
        
        # Sampling parameters - enable logprobs to get token probabilities
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=10)
        results = {}
        
        for model_name in models:
            print(f"\nTesting model: {model_name}")
            
            # Create LLM instance for each model
            llm = LLM(model=model_name, tensor_parallel_size=1)
            
            try:
                model_results = {}
                for dataset_name, dataset_path in datasets.items():
                    print(f"  Testing {dataset_name}...")
                    data = load_jsonl(dataset_path)
                    accuracy = test_dataset(llm, data, dataset_name, sampling_params)
                    model_results[dataset_name] = {
                        'accuracy': accuracy,
                        'total_samples': len(data)
                    }
                    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(data))}/{len(data)})")
                
                results[model_name] = model_results
                
            finally:
                # Clean up LLM instance and free GPU memory
                del llm
                import gc
                gc.collect()
                
                # Clear Ray object store
                try:
                    ray.internal.free_objects_in_object_store_memory()
                except:
                    pass
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        for model_name, model_results in results.items():
            print(f"\nModel: {model_name}")
            print("-" * 40)
            for dataset_name, result in model_results.items():
                accuracy = result['accuracy']
                total = result['total_samples']
                correct = int(accuracy * total)
                print(f"{dataset_name:>10}: {accuracy:.4f} ({correct}/{total})")
        
        # Save results to JSON
        with open('tmp.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to tmp.json")
        
    finally:
        # Shutdown Ray to free all resources
        ray.shutdown()
        print("\nRay shutdown completed. All resources freed.")

if __name__ == "__main__":
    main()
