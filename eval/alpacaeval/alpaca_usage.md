# 关于Alpaca Eval
对指令遵循模型(例如，ChatGPT)的评估通常需要人工交互。这既耗时又昂贵，而且很难复制。AlpacaEval是一个基于llm的自动评估，它快速、廉价、可复制，并针对20K个人工注释进行了验证。

基本思想：将GPT-4作为评判模型（annotator），对于相同指令集，挑选2个模型生成的回复中更好的那一个，从而实现自动评判。所以，我们只要用模型生成回答，交给AlpacaEval就能评判（支持我们两个模型的输出对打，也可以我们一个模型输出跟AlpacaEval的基准模型GPT4打）。

# 安装
`pip install alpaca-eval`
# 基本使用
```
export OPENAI_API_KEY=<your_api_key> 
alpaca_eval --model_outputs 'example/outputs.json' 
```
参数：

`model_outputs` :一个json文件的路径，用于将模型的输出添加到排行榜中。每个字典都应该包含`instruction` 和`output`。

`annotators_config`:这是要使用的评判模型（叫annotators，也就是GPT4）。我们建议使用weighted_alpaca_eval_gpt4_turbo (AlpacaEval 2.0的默认值)，它与我们的人类注释数据具有很高的一致性，具有较大的上下文大小，并且非常便宜。

`reference_outputs`:参考模型的输出。与`model_outputs`格式相同。默认情况下，这是AlpacaEval 2.0的gpt4_turbo。

`output_path`:保存注释和排行榜的路径。

参照指令1：
```
alpaca_eval --model_outputs 'final_eval.json' --reference_outputs 'alpaca_52k.json' --output_path 'final_vs_alpaca52k' 
```

参照指令2：
```
alpaca_eval --model_outputs 'example/outputs.json' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'
```

参照指令3：
evaluate_from_model是从模型获得输出
```
# need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs 'oasst_pythia_12b' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'      
```