import time
from vllm import LLM, SamplingParams

# --- 配置路径 (保持不变) ---
MODEL_PATH = "/backup01/DLM/model/Llama-3.1-8B-Instruct"
# MODEL_PATH = "/backup01/DLM/model/Qwen2.5-7B-Instruct"

def main():
    # 1. 初始化引擎 (保持不变)
    print(f"正在加载模型: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.9, 
        max_model_len=4096, 
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

    # 3. 准备数据 (保持不变)
    raw_queries = [
        "请用一句话解释相对论。",
        "写一首关于春天的五言绝句。",
        "Python 中的 list 和 tuple 有什么区别？",
        "鲁迅和周树人是什么关系？",
        "给初学者推荐三个健身动作。"
    ]

    print(f"\n>>> 正在准备 {len(raw_queries)} 条请求... <<<")

    # 批量构建 Prompt
    prompts_list = []
    for query in raw_queries:
        messages = [
            {"role": "system", "content": "你是一个很有用的助手。"},
            {"role": "user", "content": query}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts_list.append(text)

    # --- 【核心改动区域：串行处理】 ---
    
    print(f"开始串行处理 (One-by-One)...")
    start_time = time.time()
    
    outputs = [] # 用于收集所有结果
    
    # 使用循环，每次只生成一条，强制模型串行执行
    for i, prompt in enumerate(prompts_list):
        print(f"  -> 正在处理第 {i+1}/{len(prompts_list)} 条请求...")
        
        # 关键点：generate 中传入只有 1 个元素的列表
        # use_tqdm=False 是为了避免进度条在循环里重复刷屏
        single_output = llm.generate([prompt], sampling_params, use_tqdm=False)
        
        # 将结果存入总列表
        outputs.extend(single_output)

    end_time = time.time()
    duration = end_time - start_time

    # 5. 输出结果
    print(f"\n{'='*10} 串行处理完成 {'='*10}")
    print(f"串行处理 {len(raw_queries)} 条请求，总耗时: {duration:.2f} 秒")
    print(f"平均每条耗时: {duration / len(raw_queries):.2f} 秒 (这应该比批处理慢)\n")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"[请求 {i+1}] 用户: {raw_queries[i]}")
        print(f"[回复] 助手: {generated_text.strip()}\n")
        print("-" * 30)

if __name__ == "__main__":
    main()