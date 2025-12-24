import json
import os
from glob import glob

def get_model_scores(prefix):
    """获取指定前缀模型的四个评测成绩"""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    
    # 查找匹配前缀的目录
    dirs = glob(f"{base_path}/{prefix}*")
    
    results = {}
    for dir_path in dirs:
        model_name = os.path.basename(dir_path)
        scores = {}
        
        # 读取四个评测文件
        for metric in ["aime24", "math_oai", "minerva_math", "olympiadbench"]:
            metric_file = f"{dir_path}/{metric}_metrics.json"
            if os.path.exists(metric_file):
                with open(metric_file) as f:
                    data = json.load(f)
                    scores[metric] = data.get("mean_acc", 0)
        
        if scores:
            results[model_name] = scores
    
    return results

def print_markdown_table(prefix):
    """生成Markdown表格格式的结果"""
    results = get_model_scores(prefix)
    
    if not results:
        return f"未找到匹配 {prefix} 的模型"
    
    # 计算平均分并排序
    for model, scores in results.items():
        valid_scores = [s for s in scores.values() if s != "-"]
        results[model]["avg"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    # 按平均分排序（从大到小）
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg"], reverse=True)
    
    # 表格头
    print(f"# {prefix} 模型评测结果\n")
    print("| Model | AIME24 | MATH-OAI | Minerva Math | OlympiadBench | Avg |")
    print("|-------|--------|----------|--------------|---------------|-----|")
    
    # 表格内容
    for model, scores in sorted_results:
        aime24 = scores.get("aime24", "-")
        math_oai = scores.get("math_oai", "-")
        minerva = scores.get("minerva_math", "-")
        olympiad = scores.get("olympiadbench", "-")
        avg = f"{scores['avg']:.1f}"
        print(f"| {model} | {aime24} | {math_oai} | {minerva} | {olympiad} | {avg} |")

# 使用示例
if __name__ == "__main__":
    print_markdown_table("Llama-2-7b")
