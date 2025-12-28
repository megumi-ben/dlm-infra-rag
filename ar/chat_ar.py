import time
import os
# os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
from vllm import LLM, SamplingParams

# --- 配置路径 (保持不变) ---
# MODEL_PATH = "/root/autodl-tmp/model/Llama-3.1-8B-Instruct"
# DRAFT_MODEL_PATH = "/root/autodl-tmp/model/EAGLE-LLaMA3.1-Instruct-8B"


MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-7B-Instruct"
# DRAFT_MODEL_PATH = "/root/autodl-tmp/model/EAGLE-Qwen2.5-7B-Instruct"

# MODEL_PATH = "/root/autodl-tmp/model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-7B-Instruct-AWQ"

def main():
    # 1. 初始化引擎 (保持不变)
    print(f"正在加载模型: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        # quantization="awq",  # 显式指定 AWQ
        # dtype="auto",      # 必须设为 auto
        # speculative_config={
        #     "method": "eagle",
        #     "model": DRAFT_MODEL_PATH, # 指向这个小权重
        #     "num_speculative_tokens": 5,
        # },
        trust_remote_code=True,
        gpu_memory_utilization=0.9, 
        max_model_len=4096,  # 限制长度防止显存溢出
        tensor_parallel_size=1, 
        dtype="bfloat16"
    )
    tokenizer = llm.get_tokenizer()

    # 2. 设置采样参数 (保持不变)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # --- 【核心改动区域】 ---
    
    # 不再使用 while 循环等待输入，而是直接准备一批数据
    # 这里模拟了 5 个并发用户发送的不同请求
    raw_queries = [
        "请用一句话解释相对论。",
        "写一首关于春天的五言绝句。",
        "Python 中的 list 和 tuple 有什么区别？",
        "鲁迅和周树人是什么关系？",
        "给初学者推荐三个健身动作。"
    ]

    print(f"\n>>> 正在准备 {len(raw_queries)} 条并发请求... <<<")

    # 3. 批量构建 Prompt
    prompts_list = []
    for query in raw_queries:
        messages = [
            {"role": "system", "content": "你是一个很有用的助手。"},
            {"role": "user", "content": query}
        ]
        # 预处理每一条 prompt
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts_list.append(text)

    # 4. 【关键】批量生成 (Batch Generation)
    # vLLM 接收到一个列表时，会自动开启 Continuous Batching
    start_time = time.time()
    
    # 以前你是 llm.generate([one_prompt])，现在是传入整个列表
    outputs = llm.generate(prompts_list, sampling_params)
    
    end_time = time.time()
    duration = end_time - start_time

    # 5. 输出结果
    print(f"\n{'='*10} 处理完成 {'='*10}")
    print(f"并发处理 {len(raw_queries)} 条请求，总耗时: {duration:.2f} 秒")
    print(f"平均每条耗时: {duration / len(raw_queries):.2f} 秒 (体现了并发优势)\n")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"[请求 {i+1}] 用户: {raw_queries[i]}")
        print(f"[回复] 助手: {generated_text.strip()}\n")
        print("-" * 30)

if __name__ == "__main__":
    main()