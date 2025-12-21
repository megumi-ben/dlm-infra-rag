# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# 
# Mix版本：融合 code2.py 中的 kick-start 逻辑与 generate.py 中的三种生成函数

import torch
import numpy as np
import torch.nn.functional as F
import math
import time
import argparse
import os
from typing import Optional, Tuple
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from rag_engine import AdvancedRAGEngine
from rag2 import FlashRAGEngine
from dataset_processor import (
    process_prompt, 
    process_retrieved_kickstart_text, 
    split_tarin_test, 
    load_evaluation_metrics
)
from evaluation import compute_metrics



def add_gumbel_noise(logits, temperature):
    """
    内存优化的 Gumbel 噪声添加函数。
    仅在计算噪声的核心步骤使用 float64，极大降低显存峰值。
    """
    if temperature == 0:
        return logits

    # 1. 预先分配输出显存（使用原始精度，如 bf16/fp16），避免最后 cat 产生两份内存
    out = torch.empty_like(logits)
    
    batch_size = logits.shape[0]
    
    # 2. 逐样本（或小 Batch）处理
    for b in range(batch_size):
        # 取出一个样本，保持维度 (1, seq_len, vocab_size)
        logit_b = logits[b:b+1] 
        
        # --- 核心高精度区 START ---
        logit_b_64 = logit_b.to(torch.float64)
        
        # 生成噪声 (Float64)
        noise = torch.rand_like(logit_b_64, dtype=torch.float64)
        
        # 防止 log(0) 导致的 -inf，加一个极小 epsilon
        epsilon = 1e-64  # float64 极小值
        gumbel_noise = (-torch.log(noise + epsilon)) ** temperature
        
        # 计算结果
        result_b = logit_b_64.exp() / gumbel_noise
        
        # --- 核心高精度区 END ---

        # 立即转回原始精度并填入输出 Tensor
        out[b] = result_b.to(logits.dtype)

    return out


def get_num_transfer_tokens(mask_index, steps):
    """
    Linear Scheduler: 线性调度
    在反向过程中，区间 [0, 1] 被均匀离散化为 steps 个区间。
    由于 LLaDA 采用线性噪声调度，每一步转移的预期 token 数应保持一致。

    Args:
        mask_index: [B, seq_len] 的 bool 张量，表示 MASK 位置
        steps: 当前块的总步数
    
    Returns:
        num_transfer_tokens: [B, steps] 的张量，表示每一步需要转移的 token 数
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    if steps == 0:
        return torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_num_transfer_tokens_cosine(mask_index, steps):
    """
    Cosine Scheduler: 余弦调度
    相比线性调度，余弦调度在前期分配更多 token（快速确定结构），
    后期分配较少 token（精细调整细节）。
    
    Args:
        mask_index: [B, seq_len] 的 bool 张量，表示 MASK 位置
        steps: 当前块的总步数
    
    Returns:
        num_transfer_tokens: [B, steps] 的张量，表示每一步需要转移的 token 数
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
    
    if steps == 0:
        return torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    batch_size = mask_num.size(0)
    device = mask_index.device
    
    # 生成余弦调度的累积比例
    # cos 从 1 降到 0，对应 (1 - cos) 从 0 升到 1
    t = torch.arange(steps + 1, device=device, dtype=torch.float32) / steps  # [0, 1/steps, ..., 1]
    # 使用 (1 - cos(t * pi)) / 2 作为累积比例，这样前期增长快，后期增长慢
    cumulative_ratio = (1 - torch.cos(t * math.pi)) / 2  # [steps + 1]
    
    # 计算每个样本在每一步应该累积转移的 token 数
    cumulative_tokens = (mask_num.float() * cumulative_ratio.unsqueeze(0)).round().long()  # [B, steps+1]
    
    # 每一步转移的 token 数 = 当前累积 - 上一步累积
    num_transfer_tokens = cumulative_tokens[:, 1:] - cumulative_tokens[:, :-1]  # [B, steps]
    
    # 确保总和正确（处理舍入误差）
    total_assigned = num_transfer_tokens.sum(dim=1, keepdim=True)
    diff = mask_num - total_assigned
    # 将差值加到最后一步
    num_transfer_tokens[:, -1] += diff.squeeze(-1)
    
    return num_transfer_tokens


# def get_transfer_index(
#     logits: torch.Tensor,
#     temperature: float,
#     remasking: str,
#     mask_index: torch.Tensor,   # (B, L) bool
#     x: torch.Tensor,            # (B, L) long
#     num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
#     threshold: float = None,
# ):
#     """
#     返回:
#         x0: (B, L) long — 提议的 tokens
#         transfer_index: (B, L) bool — 哪些位置需要在此步更新
#     """
#     # 1) 采样提议 x0
#     logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#     x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

#     # # 2) 计算置信度（或随机）
#     # if remasking == "low_confidence":
#     #     p = F.softmax(logits.to(torch.float64), dim=-1)
#     #     x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
#     # elif remasking == "random":
#     #     x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
#     # else:
#     #     raise NotImplementedError(remasking)
#      # 计算模型对预测 Token x0 的置信度
#     if remasking == 'low_confidence':
#         p = F.softmax(logits, dim=-1)
#         x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
#     elif remasking == 'random':
#         x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#     else:
#         raise NotImplementedError(remasking)

#     # 只修改被 mask 的位置；其他位置保持原样，置信度设为 -inf
#     x0 = torch.where(mask_index, x0, x)

#     neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
#     confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

#     # 3) 选择要转移的位置（向量化）
#     if threshold is not None:
#         # 转移所有置信度 >= threshold 的被 mask 位置
#         transfer_index = mask_index & (confidence >= threshold)

#         # 至少转移一个 token（总是解除最高置信度的 mask）
#         max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
#         force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

#         transfer_index = transfer_index | force_mask
#         transfer_index = transfer_index & mask_index

#         return x0, transfer_index


#     # 使用每行不同的 top-k（num_transfer_tokens），完全批量化
#     if num_transfer_tokens is None:
#         raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

#     # 确保形状为 (B,) long
#     if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
#         num_transfer_tokens = num_transfer_tokens.squeeze(1)
#     num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
#     num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

#     # 按置信度降序排序
#     values, idx = torch.sort(confidence, dim=1, descending=True)

#     B, L = confidence.shape
#     # 构建一个 mask，对每行前 k[b] 列（排序后的顺序）为 True
#     cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
#     k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
#     select_sorted = cols < k_expanded                                            # (B, L) bool

#     # 将排序后的 True/False 分散回原始列顺序
#     transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
#     transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
#     transfer_index = transfer_int.bool() & mask_index  # 确保不选择未被 mask 的位置

#     # # === 分支 B: 标准 Top-K 调度 (性能优化重点) ===
#     # # 确保 num_transfer_tokens 是 (B,) 形状
#     # if num_transfer_tokens.dim() == 2:
#     #     num_transfer_tokens = num_transfer_tokens.squeeze(1)
        
#     # # 找出当前 batch 中最大的 K 值，只需对 top-k 进行操作，无需全排序
#     # max_k = num_transfer_tokens.max().item()
#     # transfer_index = torch.zeros_like(x0, dtype=torch.bool)

#     # if max_k > 0:
#     #     # 1. 仅获取 Top-K (O(L*K) << O(L log L))
#     #     _, topk_indices = torch.topk(confidence, k=max_k, dim=1) # [B, max_k]
        
#     #     # 2. 创建有效掩码 (处理 batch 中每个样本 k 不同的情况)
#     #     # range_idx: [1, max_k]
#     #     range_idx = torch.arange(max_k, device=confidence.device).unsqueeze(0) 
#     #     # valid_mask: [B, max_k], 标记哪些 topk 是有效的
#     #     valid_mask = range_idx < num_transfer_tokens.unsqueeze(1)
        
#     #     # 3. 构造 scatter 的索引
#     #     batch_indices = torch.arange(confidence.shape[0], device=confidence.device).unsqueeze(1).expand(-1, max_k)
        
#     #     # 4. 仅更新有效位置
#     #     valid_batch = batch_indices[valid_mask]
#     #     valid_token = topk_indices[valid_mask]
        
#     #     transfer_index[valid_batch, valid_token] = True

    
#     return x0, transfer_index



# 第 160-250 行（替换整个 get_transfer_index 函数）

def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
    allow_draft_transfer: bool = False,  # 新增参数：是否允许修改草稿
    draft_mask: torch.Tensor = None,     # 新增参数：标记哪些是草稿 token
):
    """
    返回:
        x0: (B, L) long — 提议的 tokens
        transfer_index: (B, L) bool — 哪些位置需要在此步更新
    
    新增参数:
        allow_draft_transfer: 是否允许修改草稿 token（默认 False 保持原有行为）
        draft_mask: (B, L) bool，标记哪些位置是草稿（True 表示是草稿）
    """
    # 1) 采样提议 x0
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) 计算置信度
    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    # ========================================================================
    # [核心修改] 决定哪些位置可以被修改
    # ========================================================================
    if allow_draft_transfer and draft_mask is not None:
        # 允许草稿转移模式：MASK + 草稿都可以被修改
        modifiable_mask = mask_index | draft_mask  # ✅ MASK 或草稿都可修改
    else:
        # 标准模式：只有 MASK 可以被修改
        modifiable_mask = mask_index
    
    # 只修改可修改的位置，其他位置保持原样
    x0 = torch.where(modifiable_mask, x0, x)
    
    # 计算置信度：只有可修改位置有有效置信度
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(modifiable_mask, x0_p, neg_inf)  # (B, L)

    # 3) 选择要转移的位置（向量化）
    if threshold is not None:
        # 转移所有置信度 >= threshold 的可修改位置
        transfer_index = modifiable_mask & (confidence >= threshold)

        # 至少转移一个 token
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & modifiable_mask  # ✅ 确保只修改允许的位置

        return x0, transfer_index

    # 使用 top-k 选择
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # 确保形状为 (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # 按置信度降序排序
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # 构建一个 mask，对每行前 k[b] 列（排序后的顺序）为 True
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    # 将排序后的 True/False 分散回原始列顺序
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & modifiable_mask  # ✅ 确保只修改允许的位置

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    """
    动态转移策略：根据 factor 调整每步转移的 token 数量
    
    Args:
        logits: 模型输出的 logits
        temperature: 采样温度
        remasking: 重 mask 策略
        mask_index: mask 位置的布尔张量
        x: 当前序列
        num_transfer_tokens: 未使用（为了接口兼容）
        factor: 动态因子
    
    Returns:
        x0: 提议的 tokens
        transfer_index: 需要转移的位置
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


# ============================================================================
#                        KICK-START 初始化逻辑封装
# ============================================================================

# @torch.no_grad()
# def initialize_with_kickstart(
#     prompt: torch.Tensor,
#     gen_length: int,
#     mask_id: int,
#     model,
#     attention_mask: Optional[torch.Tensor] = None,
#     kickstart_ids: Optional[torch.Tensor] = None,
#     kickstart_strength: float = 1.0,        # -1 表示启用自适应阈值模式
#     kickstart_threshold: float = 0.9,       # 仅当 strength=-1 时生效
#     projection_type: str = 'confidence',
#     steps: int = 128,
#     tokenizer = None,
#     verbose: bool = False
# ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    
#     device = model.device
    
#     # ========================================================================
#     # 提前执行公共操作 - 初始化画布和 Attention Mask
#     # ========================================================================
#     # A. 初始化画布 (所有分支都需要)
#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
#     x[:, :prompt.shape[1]] = prompt.clone()
    
#     # B. 补全 Attention Mask (所有分支都需要)
#     if attention_mask is not None:
#         attention_mask = torch.cat([
#             attention_mask, 
#             torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=device)
#         ], dim=-1)
    
#     # ========================================================================
#     # 分支判断: 标准生成 vs Kick-Start 热启动
#     # ========================================================================
#     # 如果没有 kickstart_ids，或者 strength >= 1.0 (全重绘)，则回退到普通生成
#     # 注意：如果 strength == -1，我们不应该在这里回退
#     if kickstart_ids is None or kickstart_strength >= 1.0:
#         # >>> 标准生成模式 (无需进一步处理) <<<
#         return x, steps, attention_mask
    
#     # ========================================================================
#     # Kick-Start 热启动逻辑
#     # ========================================================================
#     if verbose:
#         mode_str = f"Adaptive (Threshold={kickstart_threshold})" if kickstart_strength == -1 else f"Fixed (Strength={kickstart_strength:.2f})"
#         print(f"\n=== [Kick-Start] Enabled | Mode: {mode_str} ===")

#     # C. 处理草稿：填充或截断
#     kickstart_len = kickstart_ids.shape[1]
#     if kickstart_len < gen_length:
#         padding = torch.full((prompt.shape[0], gen_length - kickstart_len), mask_id, dtype=torch.long).to(device)
#         kickstart_ids_padded = torch.cat([kickstart_ids, padding], dim=1)
#     else:
#         kickstart_ids_padded = kickstart_ids[:, :gen_length]
    
#     # 将草稿填入生成区
#     x[:, prompt.shape[1]:] = kickstart_ids_padded
    
#     # D. 计算实际草稿长度 (不计 padding mask)
#     gen_part = x[:, prompt.shape[1]:]
#     is_real_draft = (gen_part != mask_id) # [B, L_gen]
#     real_draft_lens = is_real_draft.sum(dim=1)
    
#     mask_indices_list = []
    
#     # E. 准备置信度 (仅当需要时)
#     need_confidence = (kickstart_strength == -1) or (projection_type == 'confidence')
    
#     gen_confidences = None
#     if need_confidence:
#         # 此时 attention_mask 已经扩展好，直接使用
#         outputs = model(x, attention_mask=attention_mask)
#         draft_probs = F.softmax(outputs.logits, dim=-1)
#         token_confidences = torch.gather(draft_probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
#         gen_confidences = token_confidences[:, prompt.shape[1]:]
        
#         # 保护 Padding Mask 不被选中
#         gen_confidences = gen_confidences.masked_fill(~is_real_draft, float('inf'))

#     # F. 逐样本决定 Mask 哪些位置
#     total_mask_ratio = 0.0
    
#     for b in range(prompt.shape[0]):
#         n_draft = real_draft_lens[b].item()
        
#         if n_draft == 0:
#             mask_indices_list.append(torch.tensor([], dtype=torch.long, device=device))
#             total_mask_ratio += 1.0
#             continue
            
#         # 根据模式选择 Mask 策略
#         if kickstart_strength == -1:
#             # 自适应阈值模式
#             confs = gen_confidences[b]
#             low_conf_indices = torch.nonzero(confs < kickstart_threshold, as_tuple=True)[0]
#             mask_indices_list.append(low_conf_indices)
#             ratio = len(low_conf_indices) / n_draft
#             total_mask_ratio += ratio
            
#             if verbose and b == 0:
#                 print(f"   [Sample 0] Adaptive: Masked {len(low_conf_indices)}/{n_draft} ({ratio:.1%}) tokens < {kickstart_threshold}")
#         else:
#             # 固定比例模式
#             num_to_mask = int(n_draft * kickstart_strength)
            
#             if num_to_mask == 0 and kickstart_strength > 0.05:
#                 num_to_mask = 1
            
#             if num_to_mask <= 0:
#                 mask_indices_list.append(torch.tensor([], dtype=torch.long, device=device))
#                 total_mask_ratio += 0.0
#                 continue
            
#             if projection_type == 'random':
#                 valid_indices = torch.nonzero(is_real_draft[b], as_tuple=True)[0]
#                 perm = torch.randperm(len(valid_indices), device=device)[:num_to_mask]
#                 mask_indices_list.append(valid_indices[perm])
#             elif projection_type == 'confidence':
#                 # num_to_mask = n_draft-1
#                 _, mask_rel_indices = torch.topk(gen_confidences[b], k=num_to_mask, largest=False)
#                 mask_indices_list.append(mask_rel_indices)
            
#             total_mask_ratio += kickstart_strength

#     # G. 应用 Mask
#     for b, indices in enumerate(mask_indices_list):
#         if len(indices) > 0:
#             global_indices = prompt.shape[1] + indices
#             x[b, global_indices] = mask_id
            
#     # H. 步数调整
#     avg_mask_ratio = total_mask_ratio / prompt.shape[0]
#     effective_steps = int(steps * avg_mask_ratio)
    
#     if avg_mask_ratio > 0 and effective_steps == 0:
#         effective_steps = 1
        
#     if verbose:
#         print(f"[Kick-Start] Avg Mask Ratio: {avg_mask_ratio:.2%}")
#         print(f"[Kick-Start] Steps adjusted: {steps} -> {effective_steps}")
        
#     return x, effective_steps, attention_mask

# 第 340-430 行（修改 initialize_with_kickstart 函数）

@torch.no_grad()
def initialize_with_kickstart(
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    model,
    attention_mask: Optional[torch.Tensor] = None,
    kickstart_ids: Optional[torch.Tensor] = None,
    kickstart_strength: float = 1.0,
    kickstart_threshold: float = 0.9,
    projection_type: str = 'confidence',
    steps: int = 128,
    tokenizer = None,
    verbose: bool = False,
    allow_draft_transfer: bool = False,  # 新增参数
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:  # ✅ 返回 draft_mask
    
    device = model.device
    
    # A-B. 初始化画布和 Attention Mask
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    # 初始化 draft_mask (标记哪些位置是草稿)
    draft_mask = torch.zeros_like(x, dtype=torch.bool)  # ✅ 新增
    
    # 分支判断
    if kickstart_ids is None or kickstart_strength >= 1.0:
        return x, steps, attention_mask, draft_mask  # ✅ 返回 draft_mask
    
    if verbose:
        mode_str = f"Adaptive (Threshold={kickstart_threshold})" if kickstart_strength == -1 else f"Fixed (Strength={kickstart_strength:.2f})"
        transfer_mode = "允许草稿转移" if allow_draft_transfer else "草稿固定"
        print(f"\n=== [Kick-Start] Enabled | Mode: {mode_str} | {transfer_mode} ===")
    
    # C. 处理草稿
    kickstart_len = kickstart_ids.shape[1]
    if kickstart_len < gen_length:
        padding = torch.full((prompt.shape[0], gen_length - kickstart_len), mask_id, dtype=torch.long).to(device)
        kickstart_ids_padded = torch.cat([kickstart_ids, padding], dim=1)
    else:
        kickstart_ids_padded = kickstart_ids[:, :gen_length]
    
    # 将草稿填入生成区
    x[:, prompt.shape[1]:] = kickstart_ids_padded
    
    # ✅ 标记哪些位置是真实草稿（不是 padding）
    gen_part = x[:, prompt.shape[1]:]
    is_real_draft = (gen_part != mask_id)
    draft_mask[:, prompt.shape[1]:] = is_real_draft  # ✅ 标记草稿位置
    
    # D. 计算实际草稿长度
    real_draft_lens = is_real_draft.sum(dim=1)
    
    mask_indices_list = []
    
    # E. 准备置信度
    need_confidence = (kickstart_strength == -1) or (projection_type == 'confidence')
    
    gen_confidences = None
    if need_confidence:
        outputs = model(x, attention_mask=attention_mask)
        draft_probs = F.softmax(outputs.logits, dim=-1)
        token_confidences = torch.gather(draft_probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        gen_confidences = token_confidences[:, prompt.shape[1]:]
        
        # 保护 Padding Mask
        gen_confidences = gen_confidences.masked_fill(~is_real_draft, float('inf'))

    # F. 逐样本决定 Mask 哪些位置
    total_mask_ratio = 0.0
    
    for b in range(prompt.shape[0]):
        n_draft = real_draft_lens[b].item()
        
        if n_draft == 0:
            mask_indices_list.append(torch.tensor([], dtype=torch.long, device=device))
            total_mask_ratio += 1.0
            continue
        
        # 根据模式选择 Mask 策略
        if kickstart_strength == -1:
            confs = gen_confidences[b]
            low_conf_indices = torch.nonzero(confs < kickstart_threshold, as_tuple=True)[0]
            mask_indices_list.append(low_conf_indices)
            ratio = len(low_conf_indices) / n_draft
            total_mask_ratio += ratio
            
            if verbose and b == 0:
                print(f"   [Sample 0] Adaptive: Masked {len(low_conf_indices)}/{n_draft} ({ratio:.1%}) tokens < {kickstart_threshold}")
        else:
            num_to_mask = int(n_draft * kickstart_strength)
            
            if num_to_mask == 0 and kickstart_strength > 0.05:
                num_to_mask = 1
            
            if num_to_mask <= 0:
                mask_indices_list.append(torch.tensor([], dtype=torch.long, device=device))
                total_mask_ratio += 0.0
                continue
            
            if projection_type == 'random':
                valid_indices = torch.nonzero(is_real_draft[b], as_tuple=True)[0]
                perm = torch.randperm(len(valid_indices), device=device)[:num_to_mask]
                mask_indices_list.append(valid_indices[perm])
            elif projection_type == 'confidence':
                _, mask_rel_indices = torch.topk(gen_confidences[b], k=num_to_mask, largest=False)
                mask_indices_list.append(mask_rel_indices)
            
            total_mask_ratio += kickstart_strength

    # G. 应用 Mask
    for b, indices in enumerate(mask_indices_list):
        if len(indices) > 0:
            global_indices = prompt.shape[1] + indices
            x[b, global_indices] = mask_id
            # ✅ 被 MASK 的位置不再是草稿
            draft_mask[b, global_indices] = False
    
    # H. 步数调整
    avg_mask_ratio = total_mask_ratio / prompt.shape[0]
    effective_steps = int(steps * avg_mask_ratio)
    
    if avg_mask_ratio > 0 and effective_steps == 0:
        effective_steps = 1
    
    if verbose:
        print(f"[Kick-Start] Avg Mask Ratio: {avg_mask_ratio:.2%}")
        print(f"[Kick-Start] Steps adjusted: {steps} -> {effective_steps}")
        print(f"[Kick-Start] Draft tokens: {draft_mask.sum().item()} (可转移: {allow_draft_transfer})")
    
    return x, effective_steps, attention_mask, draft_mask  # ✅ 返回 draft_mask

# ============================================================================
#                    带 KICK-START 的三种生成函数
# ============================================================================

@torch.no_grad()
def generate_kickstart(
    model, prompt, 
    attention_mask=None, 
    tokenizer=None,
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    remasking='low_confidence', 
    mask_id=126336,
    threshold=None,
    factor=None,
    # Kick-start 参数
    kickstart_ids=None,
    kickstart_threshold=0.9,
    max_steps_per_block=128,
    kickstart_strength=1.0,
    projection_type='confidence',
    scheduler='cosine',
    verbose=False,
    allow_draft_transfer=False,  # ✅ 新增参数
):
    """
    LLaDA 标准生成函数 + Kick-Start 支持
    
    Args:
        model: LLaDA 模型实例
        prompt: 输入提示的 token ids
        attention_mask: 注意力掩码
        tokenizer: 分词器（用于 verbose）
        steps: 扩散生成的总步数
        gen_length: 生成序列的长度
        block_length: 块长度
        temperature: 采样温度
        remasking: 重 mask 策略
        mask_id: MASK token id
        kickstart_ids: 检索到的草稿 token ids
        kickstart_strength: 重绘强度 (0.0 - 1.0)
        projection_type: 投影策略 'random' 或 'confidence'
        scheduler: 调度器类型 'linear' 或 'cosine'
        verbose: 是否打印详细信息
    
    Returns:
        x: 生成的序列
        effective_steps: 实际运行的步数
    """
    device = model.device
    
    # === KICK-START 初始化 ===
    x, effective_steps, attention_mask, _ = initialize_with_kickstart(
        prompt=prompt,
        gen_length=gen_length,
        mask_id=mask_id,
        model=model,
        attention_mask=attention_mask,
        kickstart_ids=kickstart_ids,
        kickstart_strength=kickstart_strength,
        kickstart_threshold=kickstart_threshold,
        projection_type=projection_type,
        steps=steps,
        tokenizer=tokenizer,
        verbose=verbose
    )
    
    # === 原有的 Blockwise 推理循环 ===
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # 步数安全检查
    if effective_steps < num_blocks:
        if verbose:
            print(f"[Warning] Effective steps ({effective_steps}) too low, forcing to {num_blocks}.")
        effective_steps = num_blocks
    
    # 调整步数以整除 Block 数
    if effective_steps % num_blocks != 0:
        effective_steps = (effective_steps // num_blocks) * num_blocks
        if effective_steps == 0:
            effective_steps = num_blocks
        if verbose:
            print(f"[Warning] Adjusted steps to {effective_steps} to match block count.")
    
    effective_steps_per_block = effective_steps // num_blocks
    
    if (kickstart_strength == -1) or ((kickstart_strength <1) and (projection_type == 'confidence')):
        nfe = 1
    else:
        nfe = 0
    
    # --- 外层循环: Block (分块生成) ---
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # 根据调度器选择
        if scheduler == 'cosine':
            num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, effective_steps_per_block)
        else:  # linear
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, effective_steps_per_block)
        
        # --- 内层循环: Iterative Denoising (使用 while True 以支持提前退出) ---
        i = 0
        while True:
            mask_index = (x == mask_id)
            
            # Model Forward
            logits = model(x, attention_mask=attention_mask).logits
            
            # 忽略未来 Block 的置信度
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            
            # Transfer Index 计算（支持 threshold 和 factor）
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x,
                    num_transfer_tokens[:, i] if threshold is None else None, threshold
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor
                )
            
            # 更新 x
            x[transfer_index] = x0[transfer_index]
            
            i += 1
            nfe += 1
            
            # 提前退出：当前 Block 的所有 Mask 都已填充
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            # Safety Break
            if i > max_steps_per_block: 
                if verbose: print("   [Warning] Force breaking loop")
                break
            
            # 显存清理
            del logits, x0
        # # --- 内层循环: 改回 for 循环，移除同步检ad查 ---
        # for i in range(effective_steps_per_block):  # <--- 直接循环固定步数
        #     mask_index = (x == mask_id)
            
        #     # Model Forward
        #     logits = model(x, attention_mask=attention_mask).logits
            
        #     # 忽略未来 Block 的置信度
        #     mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            
        #     # 调用更新逻辑
        #     if factor is None:
        #         x0, transfer_index = get_transfer_index(
        #             logits, temperature, remasking, mask_index, x,
        #             num_transfer_tokens[:, i] if threshold is None else None, 
        #             threshold
        #         )
        #     else:
        #         x0, transfer_index = get_transfer_index_dynamic(
        #             logits, temperature, remasking, mask_index, x, None, factor
        #         )
        #     # 更新 x
        #     x[transfer_index] = x0[transfer_index]

    
    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache_kickstart(
    model, prompt,
    attention_mask=None,
    tokenizer=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    factor=None,
    # Kick-start 参数
    kickstart_ids=None,
    kickstart_threshold=0.9,
    max_steps_per_block=128,
    kickstart_strength=1.0,
    projection_type='confidence',
    scheduler='cosine',
    verbose=False,
    allow_draft_transfer=False,  # ✅ 新增参数
):
    """
    LLaDA Prefix Cache 生成函数 + Kick-Start 支持
    
    参数说明同 generate_kickstart
    """
    device = model.device
    
    # === KICK-START 初始化 ===
    x, effective_steps, attention_mask, _ = initialize_with_kickstart(
        prompt=prompt,
        gen_length=gen_length,
        mask_id=mask_id,
        model=model,
        attention_mask=attention_mask,
        kickstart_ids=kickstart_ids,
        kickstart_strength=kickstart_strength,
        kickstart_threshold=kickstart_threshold,
        projection_type=projection_type,
        steps=steps,
        tokenizer=tokenizer,
        verbose=verbose
    )
    
    # === Blockwise 推理循环（带 Prefix Cache） ===
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # 步数安全检查
    if effective_steps < num_blocks:
        if verbose:
            print(f"[Warning] Effective steps ({effective_steps}) too low, forcing to {num_blocks}.")
        effective_steps = num_blocks
    
    if effective_steps % num_blocks != 0:
        effective_steps = (effective_steps // num_blocks) * num_blocks
        if effective_steps == 0:
            effective_steps = num_blocks
        if verbose:
            print(f"[Warning] Adjusted steps to {effective_steps} to match block count.")
    
    effective_steps_per_block = effective_steps // num_blocks
    
    if (kickstart_strength == -1) or ((kickstart_strength <1) and (projection_type == 'confidence')):
        nfe = 1
    else:
        nfe = 0
    
    # --- 外层循环: Block ---
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        
        # 根据调度器选择
        if scheduler == 'cosine':
            num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, effective_steps_per_block)
        else:  # linear
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, effective_steps_per_block)
        
        # 第一次前向：生成 KV Cache
        output = model(x, attention_mask=attention_mask, use_cache=True)
        past_key_values = output.past_key_values
        
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        
        # 支持 threshold 和 factor
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits, temperature, remasking, mask_index, x,
                num_transfer_tokens[:, 0] if threshold is None else None, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                output.logits, temperature, remasking, mask_index, x, None, factor
            )
        
        x[transfer_index] = x0[transfer_index]
        
        # 裁剪 KV Cache 到当前块开始位置
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        # --- 内层循环：使用 Cache 进行迭代（while True 支持提前退出）---
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
            
            # 支持 threshold 和 factor
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index,
                    x[:, current_block_start:],
                    num_transfer_tokens[:, i] if threshold is None else None, threshold
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index,
                    x[:, current_block_start:], None, factor
                )
            
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1
            # Safety Break
            if i > max_steps_per_block: 
                if verbose: print("   [Warning] Force breaking loop")
                break
    
    return x, nfe


# @torch.no_grad()
# def generate_with_dual_cache_kickstart(
#     model, prompt,
#     attention_mask=None,
#     tokenizer=None,
#     steps=128,
#     gen_length=128,
#     block_length=32,
#     temperature=0.,
#     remasking='low_confidence',
#     mask_id=126336,
#     threshold=None,
#     factor=None,
#     # Kick-start 参数
#     kickstart_ids=None,
#     kickstart_strength=1.0,
#     kickstart_threshold=0.95,
#     max_steps_per_block=128,
#     projection_type='confidence',
#     scheduler='cosine',
#     verbose=False
# ):
#     """
#     LLaDA Dual Cache 生成函数 + Kick-Start 支持
    
#     参数说明同 generate_kickstart
#     """
#     device = model.device
    
#     # === KICK-START 初始化 ===
#     x, effective_steps, attention_mask = initialize_with_kickstart(
#         prompt=prompt,
#         gen_length=gen_length,
#         mask_id=mask_id,
#         model=model,
#         attention_mask=attention_mask,
#         kickstart_ids=kickstart_ids,
#         kickstart_strength=kickstart_strength,
#         kickstart_threshold=kickstart_threshold,
#         projection_type=projection_type,
#         steps=steps,
#         tokenizer=tokenizer,
#         verbose=verbose
#     )
    
#     # === Blockwise 推理循环（带 Dual Cache） ===
#     B = prompt.shape[0]
#     Lp = int(prompt.shape[1])
    
#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length
    
#     # 步数安全检查
#     if effective_steps < num_blocks:
#         if verbose:
#             print(f"[Warning] Effective steps ({effective_steps}) too low, forcing to {num_blocks}.")
#         effective_steps = num_blocks
    
#     if effective_steps % num_blocks != 0:
#         effective_steps = (effective_steps // num_blocks) * num_blocks
#         if effective_steps == 0:
#             effective_steps = num_blocks
#         if verbose:
#             print(f"[Warning] Adjusted steps to {effective_steps} to match block count.")
    
#     steps_per_block = effective_steps // num_blocks
    
#     if (kickstart_strength == -1) or ((kickstart_strength <1) and (projection_type == 'confidence')):
#         nfe = 1
#     else:
#         nfe = 0
    
#     # --- 外层循环: Block ---
#     for nb in range(num_blocks):
#         s = Lp + nb * block_length
#         e = s + block_length
        
#         block_mask_index = (x[:, s:e] == mask_id)
        
#         # 根据调度器选择
#         if scheduler == 'cosine':
#             num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps_per_block)
#         else:  # linear
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
#         # 1) 在完整前缀上预热 KV-cache
#         out_full = model(x, attention_mask=attention_mask, use_cache=True)
#         past_key_values = out_full.past_key_values
#         nfe += 1
        
#         # 构建 replace_position 张量，指示块范围（静态切片）
#         replace_position = torch.zeros_like(x, dtype=torch.bool)
#         replace_position[:, s:e] = True
        
#         # Step 0: 在完整 logits 上进行初始转移
#         global_mask_index = (x == mask_id)
#         global_mask_index[:, e:] = False
        
#         # 支持 threshold 和 factor
#         if factor is None:
#             x0, transfer_index = get_transfer_index(
#                 out_full.logits, temperature, remasking, global_mask_index, x,
#                 num_transfer_tokens[:, 0] if threshold is None else None, threshold
#             )
#         else:
#             x0, transfer_index = get_transfer_index_dynamic(
#                 out_full.logits, temperature, remasking, global_mask_index, x, None, factor
#             )
        
#         # 使用 torch.where 进行原地更新
#         x = torch.where(transfer_index, x0, x)
        
#         # 2) 半自回归细化（while True 支持提前退出）
#         i = 1
#         while True:
#             if (x[:, s:e] == mask_id).sum() == 0:
#                 break
            
#             logits_blk = model(
#                 x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
#             ).logits
            
#             mask_blk = (x[:, s:e] == mask_id)
            
#             # 支持 threshold 和 factor
#             if factor is None:
#                 x0_blk, transfer_idx_blk = get_transfer_index(
#                     logits_blk, temperature, remasking, mask_blk, x[:, s:e],
#                     num_transfer_tokens[:, i] if threshold is None else None, threshold
#                 )
#             else:
#                 x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
#                     logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
#                 )
            
#             # 合并回 x[:, s:e] 使用 torch.where
#             blk_old = x[:, s:e]
#             blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
#             x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            
#             nfe += 1
#             i += 1
#             # Safety Break
#             if i > max_steps_per_block: 
#                 if verbose: print("   [Warning] Force breaking loop")
#                 break
    
#     return x, nfe


@torch.no_grad()
def generate_with_dual_cache_kickstart(
    model, prompt,
    attention_mask=None,
    tokenizer=None,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    factor=None,
    # Kick-start 参数
    kickstart_ids=None,
    kickstart_strength=1.0,
    kickstart_threshold=0.95,
    max_steps_per_block=128,
    projection_type='confidence',
    scheduler='cosine',
    verbose=False,
    allow_draft_transfer=False,  # ✅ 新增参数
):
    """
    LLaDA Dual Cache 生成函数 + Kick-Start 支持
    
    参数说明同 generate_kickstart
    """
    device = model.device
    
    # === KICK-START 初始化 ===
    x, effective_steps, attention_mask, draft_mask = initialize_with_kickstart(
        prompt=prompt,
        gen_length=gen_length,
        mask_id=mask_id,
        model=model,
        attention_mask=attention_mask,
        kickstart_ids=kickstart_ids,
        kickstart_strength=kickstart_strength,
        kickstart_threshold=kickstart_threshold,
        projection_type=projection_type,
        steps=steps,
        tokenizer=tokenizer,
        verbose=verbose,
        allow_draft_transfer=allow_draft_transfer,  # ✅ 传递参数
    )
    
    # === Blockwise 推理循环（带 Dual Cache） ===
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # 步数安全检查
    if effective_steps < num_blocks:
        if verbose:
            print(f"[Warning] Effective steps ({effective_steps}) too low, forcing to {num_blocks}.")
        effective_steps = num_blocks
    
    if effective_steps % num_blocks != 0:
        effective_steps = (effective_steps // num_blocks) * num_blocks
        if effective_steps == 0:
            effective_steps = num_blocks
        if verbose:
            print(f"[Warning] Adjusted steps to {effective_steps} to match block count.")
    
    steps_per_block = effective_steps // num_blocks
    
    if (kickstart_strength == -1) or ((kickstart_strength <1) and (projection_type == 'confidence')):
        nfe = 1
    else:
        nfe = 0
    
    # --- 外层循环: Block ---
    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        
        block_mask_index = (x[:, s:e] == mask_id)
        
        # 根据调度器选择
        if scheduler == 'cosine':
            num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps_per_block)
        else:  # linear
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # 1) 在完整前缀上预热 KV-cache
        out_full = model(x, attention_mask=attention_mask, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1
        
        # 构建 replace_position 张量，指示块范围（静态切片）
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True
        
        # Step 0: 在完整 logits 上进行初始转移
        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False
        

        
        # 支持 threshold 和 factor
        if factor is None:
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x,
                num_transfer_tokens[:, 0] if threshold is None else None, threshold,
                allow_draft_transfer=allow_draft_transfer,  # ✅ 新增
                draft_mask=draft_mask,  # ✅ 全局级别，维度匹配
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )
        
        # 使用 torch.where 进行原地更新
        x = torch.where(transfer_index, x0, x)
        # ✅ 更新 draft_mask：已转移的位置不再是草稿
        if allow_draft_transfer:
            draft_mask[transfer_index] = False
        
        # 2) 半自回归细化（while True 支持提前退出）
        i = 1
        while True:
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits
            
            mask_blk = (x[:, s:e] == mask_id)
            
            # ✅ 关键修复：切片 draft_mask 到块级别
            draft_mask_blk = draft_mask[:, s:e]  # [B, block_length]
            
            # 支持 threshold 和 factor
            if factor is None:
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e],
                    num_transfer_tokens[:, i] if threshold is None else None, threshold,
                    allow_draft_transfer=allow_draft_transfer,  # ✅ 新增
                    draft_mask=draft_mask_blk,  # ✅ 使用块级别的 draft_mask
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )
            
            # 合并回 x[:, s:e] 使用 torch.where
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            # ✅ 更新 draft_mask：已转移的位置不再是草稿
            if allow_draft_transfer:
                # 创建一个全局的 transfer_index
                transfer_index_global = torch.zeros_like(draft_mask, dtype=torch.bool)
                transfer_index_global[:, s:e] = transfer_idx_blk
                draft_mask = draft_mask & ~transfer_index_global
            nfe += 1
            i += 1
            # Safety Break
            if i > max_steps_per_block: 
                if verbose: print("   [Warning] Force breaking loop")
                break
    
    return x, nfe


# ============================================================================
#                              测试代码 & 实验主函数
# ============================================================================

def main():
    """
    完整的 DLM Kick-Start 实验主函数
    支持三种生成模式：generate_kickstart, generate_with_prefix_cache_kickstart, generate_with_dual_cache_kickstart
    """
    
    # --- Argparse 设置 ---
    parser = argparse.ArgumentParser(description="运行 DLM Kick-Start 实验 (使用 mix.py 中的新函数)")
    
    # 基础配置
    parser.add_argument('--temperature', type=float, default=0, help='采样温度')
    parser.add_argument('--num_test_examples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    
    # 生成参数
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--steps', type=int, default=128, help='扩散步数')
    parser.add_argument('--block_length', type=int, default=32, help='Block 长度')
    parser.add_argument('--remasking', type=str, default='low_confidence', 
                        choices=['low_confidence', 'random'], help='重掩码策略')
    
    # Fast-dLLM 加速参数
    parser.add_argument('--threshold', type=float, default=None, help='置信度阈值 (并行生成)')
    parser.add_argument('--factor', type=float, default=None, help='动态因子 (get_transfer_index_dynamic)')
    parser.add_argument('--dynamic_gen_length', action='store_true', 
                        help='根据草稿长度动态调整 gen_length（否则使用固定 gen_length）')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/model/LLaDA-8B-Instruct', 
                        help='LLM 模型路径')
    parser.add_argument('--embed_model', type=str, 
                        default='/root/autodl-tmp/model/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620', 
                        help='Embedding 模型路径')
    parser.add_argument('--rerank_model', type=str, 
                        default='/root/autodl-tmp/model/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', 
                        help='Reranker 模型路径')
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='mmlu', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--save_result', action='store_true', help='是否保存结果文件')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='用于训练/测试划分的最大数据比例（如0.1表示只用10%数据）')
    
    # [核心实验参数]
    parser.add_argument('--test_top_k', type=int, default=1, 
                        help='要测试的检索排名深度')
    parser.add_argument('--rag_engine', type=str, default='ours', 
                        help='RAG实现引擎, 支持 ours 和 flashrag')
    
    # [是否允许选择题填充预留选项位置]
    parser.add_argument('--allow_draft_transfer', action='store_true', help='是否允许草稿token被转移')
    parser.add_argument('--option_padding', type=int, default=0, help='强制占位符token数量（如用于选择题答案前的MASK数）')
    
    # [新增：生成模式选择]
    parser.add_argument('--generation_mode', type=str, default='standard', 
                        choices=['standard', 'prefix_cache', 'dual_cache'],
                        help='生成模式: standard (标准), prefix_cache (前缀缓存), dual_cache (双缓存)')
    parser.add_argument('--projection_type', type=str, default='confidence',
                        choices=['random', 'confidence'],
                        help='投影策略: random (随机) 或 confidence (置信度)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='调度器类型: linear (线性) 或 cosine (余弦)')
    
    args = parser.parse_args()
    
    # --- 打印实验配置 ---
    print(f"\n" + "=" * 80)
    print(f"{'MIX.PY - Kick-Start 实验':^80}")
    print("=" * 80)
    print(f"数据集: {args.dataset.upper()} | 样本数: {args.num_test_examples} | Top-K: {args.test_top_k}")
    print(f"生成模式: {args.generation_mode.upper()}")
    print(f"生成参数: steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length}")
    print(f"投影策略: {args.projection_type.upper()} | 调度器: {args.scheduler.upper()}")
    print(f"重掩码: {args.remasking} | 温度: {args.temperature}")
    print(f"加速参数: threshold={args.threshold}, factor={args.factor}")
    print(f"模型: {args.model_path}")
    print("=" * 80 + "\n")
    
    device = 'cuda'
    
    
    
    # 1. 加载模型
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    model = LLaDAModelLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    print("模型加载完毕。")
    
    # 2. 准备数据
    train_prompts, train_targets, test_prompts, test_targets = split_tarin_test(args.dataset, args.data_path, args.data_ratio)
    print(f"数据划分完成: Train(RAG库)={len(train_prompts)}, Test(评估)={len(test_prompts)}")
    
    # 3. 构建 RAG
    print("正在构建 RAG 知识库...")
    if args.rag_engine == 'flashrag':
        print("RAG引擎: FlashRAG ...")
        rag = FlashRAGEngine()
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    else:
        print("RAG引擎: ours ...")
        rag = AdvancedRAGEngine(embed_model=args.embed_model, rerank_model=args.rerank_model, device=device)
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    print("RAG 知识库构建完毕。")
    
    # --- 实验循环配置 ---
    # 使用命令行参数
    total_steps_config = args.steps
    block_length_config = args.block_length
    gen_length_config = args.gen_length
    batch_size = args.batch_size
    
    # 截断样本数
    if args.num_test_examples > len(test_prompts):
        args.num_test_examples = len(test_prompts)
    
    # 定义实验网格
    # steps = [-128,0,1,2,128]
    steps = [-128,0,1,2,4,8,16,32,64,96,128]
    # steps = [-128,0,112,120,125,128]


    strengths_to_test = [float(si/128) for si in steps]
    
    strengths_to_test[0]=int(-1)
    # [0.01, 0.02,0.04,0.00.1, 0.2,0.4]
    ranks_to_test = list(range(args.test_top_k))
    
    # 初始化统计容器
    stats_storage = {}
    for strength in strengths_to_test:
        for rank in ranks_to_test:
            stats_storage[(strength, rank)] = {
                'total_time': 0, 
                'total_steps_run': 0,
                'predictions': [], 
                'references': [] 
            }
    
    # 选择生成函数
    if args.generation_mode == 'standard':
        generate_fn = generate_kickstart
        print(f"\n[使用生成函数] generate_kickstart (标准模式)\n")
    elif args.generation_mode == 'prefix_cache':
        generate_fn = generate_with_prefix_cache_kickstart
        print(f"\n[使用生成函数] generate_with_prefix_cache_kickstart (前缀缓存模式)\n")
    else:  # dual_cache
        generate_fn = generate_with_dual_cache_kickstart
        print(f"\n[使用生成函数] generate_with_dual_cache_kickstart (双缓存模式)\n")
    
    # GPU 预热
    print("[GPU 预热]...")
    if test_prompts:
        warmup_inputs = tokenizer([test_prompts[0]], add_special_tokens=False, padding=True, return_tensors="pt")
        _, _ = generate_fn(
            model, 
            warmup_inputs['input_ids'].to(device), 
            attention_mask=warmup_inputs['attention_mask'].to(device),
            steps=min(32, args.steps),  # 使用较小的步数预热
            gen_length=min(32, args.gen_length), 
            block_length=min(32, args.block_length),
            temperature=args.temperature,
            remasking=args.remasking,
            threshold=args.threshold,
            factor=args.factor,
            kickstart_ids=None, 
            kickstart_strength=1.0,
            scheduler=args.scheduler,
            verbose=False 
        )
    print("[GPU 预热] 完成\n")
    
    # --- 批量推理开始 ---
    num_batches = (args.num_test_examples + batch_size - 1) // batch_size
    rag_total_time = 0.0
    rag_total_samples = 0
    emb_search_time = 0.0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, args.num_test_examples)
        current_batch_size = batch_end - batch_start
        
        batch_prompts = test_prompts[batch_start:batch_end]
        batch_targets = test_targets[batch_start:batch_end]
        
        print(f"Processing Batch {batch_idx+1}/{num_batches} (Samples {batch_start+1}-{batch_end})...")
        
        # =========================================================
        # Step A: 批量执行 Top-K 检索
        # =========================================================
        start_search = time.perf_counter()
        
        batch_retrieval_results, emb_search_elapsed = rag.batch_search(
            batch_prompts,
            top_k_retrieve=args.test_top_k,
            top_n_rerank=args.test_top_k,
            batch_size=batch_size
        )
        emb_search_time += emb_search_elapsed
        
        elapsed_search = time.perf_counter() - start_search
        rag_total_time += elapsed_search
        rag_total_samples += len(batch_prompts)
        print(f"\033[1;32m--- 知识库检索耗时: {elapsed_search:.2f}s ---\033[0m")
        
        # =========================================================
        # Step B: 准备模型输入 (Prompt Tokenization)
        # =========================================================
        batch_messages = [{"role": "user", "content": process_prompt(pt, args.dataset)} for pt in batch_prompts]
        batch_formatted_prompts = [
            tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False) 
            for msg in batch_messages
        ]
        
        encoded_inputs = tokenizer(batch_formatted_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        
        # 缓存 baseline (1.0) 的结果
        baseline_1_0_texts = None
        baseline_1_0_steps = 0
        baseline_1_0_time = 0
        
        # =========================================================
        # Step C: 双重循环 (Strength x Rank)
        # =========================================================
        for strength in strengths_to_test:
            for rank in ranks_to_test:
                
                # --- 优化: Strength 1.0 结果与 Rank 无关 ---
                if strength == 1.0 and baseline_1_0_texts is not None:
                    generated_texts = baseline_1_0_texts
                    steps_run = baseline_1_0_steps
                    time_cost = baseline_1_0_time
                    
                    stats_storage[(strength, rank)]['total_time'] += time_cost
                    stats_storage[(strength, rank)]['total_steps_run'] += steps_run * current_batch_size
                    stats_storage[(strength, rank)]['predictions'].extend(generated_texts)
                    stats_storage[(strength, rank)]['references'].extend(batch_targets)
                    continue
                
                # --- 1. 提取当前 Rank 的草稿 ---
                batch_kickstart_texts = []
                for res_list in batch_retrieval_results:
                    if res_list and len(res_list) > rank:
                        batch_kickstart_texts.append(res_list[rank]['target'])
                    else:
                        batch_kickstart_texts.append("")
                
                # --- 2. 处理草稿并 Tokenize ---
                processed_drafts = [process_retrieved_kickstart_text(t, args.dataset) for t in batch_kickstart_texts]
                kickstart_encoded = tokenizer(processed_drafts, add_special_tokens=False, padding=True, return_tensors="pt")
                kickstart_ids = kickstart_encoded['input_ids'].to(device)
                
                num_placeholders = args.option_padding
                if num_placeholders > 0 and args.dataset in ['arc','banking77','medqa']:
                    # ============================
                    # [修改开始]：插入强制占位符 (Hard Masking)
                    # ============================
                    # 设定你想要预留的 Token 数量。
                    # 例如：如果你希望模型生成 "The answer is A."，可能需要 4-5 个 token。
                    # 如果只是 "A"，可能只需要 1-2 个 token（考虑到可能产生的前缀空格）。
                    mask_id = 126336  # LLaDA 的 Mask ID

                    # 创建全为 Mask 的 Tensor [Batch, num_placeholders]
                    placeholder_tensor = torch.full(
                        (kickstart_ids.shape[0], num_placeholders), 
                        mask_id, 
                        dtype=kickstart_ids.dtype, 
                        device=device
                    )
                    # 拼接到草稿前面：[Mask, Mask, ..., Mask, Draft_Token_1, Draft_Token_2...]
                    kickstart_ids = torch.cat([placeholder_tensor, kickstart_ids], dim=1)
                
                # 根据参数决定是否动态调整 gen_length
                if args.dynamic_gen_length:
                    real_gen_length = max(64,min((kickstart_ids.shape[1] // block_length_config + 1) * block_length_config, 256))
                else:
                    real_gen_length = gen_length_config
                
                print(f"生成长度：{real_gen_length}")
                
                # --- 3. 执行生成 ---
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                
                if strength == 0.0:
                    # 纯 RAG 模式
                    generated_texts = batch_kickstart_texts
                    steps_run = 0
                else:
                    # DLM 推理模式（使用选定的生成函数）
                    infer_start = time.perf_counter()
                    out_tensor, steps_run = generate_fn(
                        model=model,
                        prompt=input_ids,
                        attention_mask=attention_mask,
                        tokenizer=tokenizer,
                        steps=total_steps_config,
                        gen_length=real_gen_length,
                        block_length=block_length_config,
                        temperature=args.temperature,
                        remasking=args.remasking,
                        threshold=args.threshold,
                        factor=args.factor,
                        # Kick-start 参数
                        kickstart_ids=kickstart_ids,
                        kickstart_strength=strength,
                        projection_type=args.projection_type,
                        scheduler=args.scheduler,
                        verbose=(batch_idx == 0 and rank == 0 and strength > 0.0),
                        allow_draft_transfer=args.allow_draft_transfer
                    )
                    infer_end = time.perf_counter() - infer_start
                    print(f"\033[1;32m--- 强度: {strength}, 本次generate耗时: {infer_end:.2f}s, 单次nfe: {steps_run}---\033[0m")
                    generated_texts = tokenizer.batch_decode(out_tensor[:, input_ids.shape[1]:], skip_special_tokens=True)
                
                torch.cuda.synchronize(device)
                time_cost = time.perf_counter() - t0
                
                # --- 4. 存储结果 ---
                stats_storage[(strength, rank)]['total_time'] += time_cost
                stats_storage[(strength, rank)]['total_steps_run'] += steps_run * current_batch_size
                stats_storage[(strength, rank)]['predictions'].extend(generated_texts)
                stats_storage[(strength, rank)]['references'].extend(batch_targets)
                
                # 缓存 baseline
                if strength == 1.0 and baseline_1_0_texts is None:
                    baseline_1_0_texts = generated_texts
                    baseline_1_0_steps = steps_run
                    baseline_1_0_time = time_cost
        
        # === 清理 ===
        del input_ids, attention_mask, encoded_inputs
        torch.cuda.empty_cache()
    
    # --- 最终评估阶段 ---
    print("\n" + "=" * 90)
    print(f"{'全维度评估结果 (Strength x Rank)':^90}")
    print("=" * 90)
    
    metric_objs = load_evaluation_metrics(args.dataset)  # 这一步再加载减少前面显存
    final_results = {}
    all_metric_keys = set()
    
    for key, stats in stats_storage.items():
        s, r = key
        if args.num_test_examples > 0:
            print(f"正在评估: Strength {s*100:.0f}% - Rank {r} ...")
            metrics = compute_metrics(stats['predictions'], stats['references'], args.dataset, metric_objs, device=device)
            
            row = {
                'Time(s)': stats['total_time'] / args.num_test_examples,
                'Steps': stats['total_steps_run'] / args.num_test_examples
            }
            row.update(metrics)
            final_results[key] = row
            all_metric_keys.update(metrics.keys())
            
            # 保存到文件
            if args.save_result:
                os.makedirs("./results", exist_ok=True)
                fname = f"./results/{args.dataset}_{args.generation_mode}_str{int(s*100)}_rank{r}.txt"
                with open(fname, 'w', encoding='utf-8') as f:
                    for line in stats['predictions']:
                        f.write(line.replace('\n', ' ') + '\n')
    
    # --- 打印超级表格 ---
    sorted_keys = sorted(final_results.keys(), key=lambda x: (-x[0], x[1]))
    metric_headers = sorted(list(all_metric_keys))
    headers = ['Strength', 'Rank', 'Time(s)', 'Steps'] + metric_headers
    col_w = 12
    
    def print_row(vals, is_header=False):
        row_s = "│ " + " │ ".join([f"{str(v):^{col_w}}" for v in vals]) + " │"
        print(row_s)
        if is_header:
            print("├" + "┼".join(["─" * (col_w + 2) for _ in vals]) + "┤")
    
    print("\n┌" + "┬".join(["─" * (col_w + 2) for _ in headers]) + "┐")
    print_row(headers, is_header=True)
    
    curr_str = -1
    for s, r in sorted_keys:
        if s != curr_str and curr_str != -1:
            pass
        curr_str = s
        
        data = final_results[(s, r)]
        cells = [f"{s*100:.0f}%", f"Top-{r+1}"]
        cells.append(f"{data.get('Time(s)',0):.3f}")
        cells.append(f"{data.get('Steps',0):.1f}")
        
        for mh in metric_headers:
            val = data.get(mh, 0)
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        
        print_row(cells)
    
    print("└" + "┴".join(["─" * (col_w + 2) for _ in headers]) + "┘")
    
    # RAG 统计
    if rag_total_samples > 0:
        print(f"\n\033[1;35m[RAG] 检索总耗时: {rag_total_time:.2f} 秒，平均每样本: {rag_total_time / rag_total_samples:.4f} 秒\033[0m")
        print(f"\033[1;35m[RAG] 向量数据库search总耗时: {emb_search_time:.3f} 秒，平均每样本: {emb_search_time / rag_total_samples:.4f} 秒\033[0m")
    
    print("\n实验结束。")


if __name__ == '__main__':
    import time
    start_time = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start_time
    print(f"\033[1;32m--- 实验总耗时: {elapsed_time:.2f}s ---\033[0m")
