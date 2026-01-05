import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 模型加载 ---
# model_id = "/backup01/DLM/model/Llama-3.1-8B-Instruct" 
model_id = "/backup01/DLM/model/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 【修正步骤 1】: 如果模型没有 pad_token，强制指定为 eos_token
# 这能解决 "Setting pad_token_id to eos_token_id" 的警告
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # 某些模型 resize 可能会报错，Llama/Qwen 一般不需要，但为了保险：
    # model.resize_token_embeddings(len(tokenizer)) 

print("\n>>> 开始对话 (输入 'exit' 退出) <<<\n")

while True:
    query = input("User: ")
    if query.strip().lower() in ["exit", "quit", "q"]:
        break
    if not query.strip():
        continue

    messages = [
        {"role": "system", "content": "你是一个很有用的助手。"},
        {"role": "user", "content": query}
    ]

    # 获取 input_ids
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 【修正步骤 2】: 手动创建 attention_mask
    # 逻辑：如果不等于 pad_token，就是有效内容(1)，否则是填充(0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # 模型生成
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,       # 【显式传入】解决 mask 警告
        pad_token_id=tokenizer.pad_token_id, # 【显式传入】解决 pad 警告
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
    )

    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    print(f"Assistant: {response}\n")
    print("-" * 30)