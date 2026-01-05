# note

chat.py 是和自回归模型对话的最简单示例（qwen、llama）
vllm_chat.py 是使用vllm框架和大模型简单的示例.
parallel_vllm_chat.py 是使用vllm框架和大模型并行批量对话的示例
serial_vllm_chat.py 是使用vllm框架和大模型逐一串行对话的示例
mix_ar.py 是主流程，加载模型使用数据集进行评测，最后计算指标。

重点是 parallel_vllm_chat 以及 mix_ar

## 特性

可以使用flashinfer

```python
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
```

可以使用eagle投机解码

```python
# DRAFT_MODEL_PATH = "/backup01/DLM/model/EAGLE-LLaMA3.1-Instruct-8B"
speculative_config={
    "method": "eagle",
    "model": DRAFT_MODEL_PATH, # 指向这个小权重
    "num_speculative_tokens": 5,
},
```

可以使用 awq 权重量化版本

```python
quantization="awq",  # 显式指定 AWQ
dtype="auto",      # 必须设为 auto

# MODEL_PATH = "/backup01/DLM/model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# MODEL_PATH = "/backup01/DLM/model/Qwen2.5-7B-Instruct-AWQ"
```

