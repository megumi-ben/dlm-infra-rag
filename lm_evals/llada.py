import torch
import numpy as np
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    计算当前块内每个样本每一步具体要填充几个MASK
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)  # 统计每个样本当前块几个MASK

    base = mask_num // steps  # 每一步基础填充几个MASK
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):  # 将余数分配到前几步（每步多填 1 个）
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens  # [batch, steps]，每个样本每一步具体要填充几个MASK


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: 是否将 EOS token 的 logits 设置为-inf
        confidence_eos_eot_inf: 是否将 EOS 和 EoT token 的 confidence 设置为-inf
    '''
    # 前面长度是prompt，后面是mask，推理阶段后面部分直接就是mask，不涉及什么随机掩码（训练阶段）
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks  # 分摊到每一块的步数

    for num_block in range(num_blocks):  # 块间串行
        # 形状为 [batch_size, block_length] 的布尔矩阵，标志是否 [MASK]
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits  # 前向传播
                # logits形状是[batch_size,seq_len,vocab_size],其中seq_len = prompt长度 + gen_length

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # 取样概率最大的，[batch_size,seq_len]
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # 计算置信度 x0_p
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # 只在 MASK 位置使用预测值，非 MASK 位置保持原值
            x0 = torch.where(mask_index, x0, x)
            # 只在 MASK 位置保留置信度，非 MASK 位置设为负无穷
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):  # 遍历 batch
                # 选择置信度top-k位置
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]  # 用x0填充x
            # Top-k 填充其实等价于隐式的重掩码了，但还是不是显式

    return x


def main():
    device = 'cuda'
    # model_path = 'GSAI-ML/LLaDA-8B-Instruct' # 或者你的本地路径
    model_path = os.path.expanduser('~/LLaDA-8B-Instruct')  # 自动转为绝对路径
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 理论填充方向左右都行，但是左填充更适合DLM，方便对齐prompt和mask部分的边界
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # [PAD] 应当和 [MASK] 不同 id
    assert tokenizer.pad_token_id != 126336

    print("=== LLaDA 单条交互模式（输入 q 退出）===")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break

        # 构造 prompt
        message = {"role": "user", "content": user_input}
        # 如果同时问多个问题，一定要分开apply_chat_template，虽然传入数组，但是每个apply_chat_template相当于一个窗口，数组内是多轮对话，只会有一个回答
        prompt = tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)

        encoded_outputs = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        # 这里是填充掩码，主要是区分真实token和填充的[PAD]，注意区别和mask掩码的区别
        attention_mask = encoded_outputs['attention_mask'].to(device)

        out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Assistant: {output}\n" + '-' * 50)

if __name__ == '__main__':
    main()