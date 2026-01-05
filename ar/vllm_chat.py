import os
from vllm import LLM, SamplingParams

# --- 配置路径 ---
# 替换为你实际的模型路径
MODEL_PATH = "/backup01/DLM/model/Llama-3.1-8B-Instruct"
# MODEL_PATH = "/backup01/DLM/model/Qwen2.5-7B-Instruct"

def main():
    # 1. 初始化引擎
    # gpu_memory_utilization=0.9: 预占90%显存作为KV Cache (vLLM的核心特性)
    # trust_remote_code=True: 某些模型(如Qwen)需要
    print(f"正在加载模型: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.9, 
        max_model_len=4096,  # 上下文不要太大，会爆显存
        tensor_parallel_size=1, # 单卡设为1
        dtype="bfloat16"        # 对应你之前的 torch.bfloat16
    )
    
    # 获取 tokenizer 用于处理对话模板
    tokenizer = llm.get_tokenizer()

    # 2. 设置采样参数
    # 对应你原来的 generate 参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id] # 遇到结束符停止
    )

    print("\n>>> vLLM 加载完成，开始对话 (输入 'exit' 退出) <<<\n")

    while True:
        query = input("User: ")
        if query.strip().lower() in ["exit", "q"]:
            break
        if not query.strip():
            continue

        # 3. 构建 Prompt (关键步骤)
        # vLLM 推荐直接输入处理好的 String，而不是 input_ids
        messages = [
            {"role": "system", "content": "你是一个很有用的助手。"},
            {"role": "user", "content": query}
        ]
        
        # apply_chat_template(tokenize=False) 会把 list 拼成字符串
        # 例如: "<|im_start|>system\n...<|im_start|>user\n..."
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. 生成 (Inference)
        # vLLM 的 generate 接收列表，这里我们传一个 [prompt]
        outputs = llm.generate([prompt], sampling_params)

        # 5. 解析结果
        generated_text = outputs[0].outputs[0].text
        print(f"Assistant: {generated_text}\n")
        print("-" * 30)

if __name__ == "__main__":
    main()