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
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    nfe: Optional[int] = None 


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        if is_torchdynamo_compiling():
            return
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            warnings.warn(
                f"Using default max_length ({generation_config.max_length}). Recommend setting max_new_tokens.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            raise ValueError(f"Input length {input_ids_length} >= max_length {generation_config.max_length}.")

    def _prepare_generated_length(self, generation_config, has_default_max_length, input_ids_length):
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_pos = getattr(self.config, "max_position_embeddings", None)
                if max_pos is not None:
                    generation_config.max_length = min(generation_config.max_length, max_pos)
        return generation_config

    def _prepare_generation_config(self, generation_config, **kwargs):
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)
        return generation_config

    def _prepare_special_tokens(self, generation_config, device=None):
        def _tensor_or_none(token, device=None):
            if token is None: return token
            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor): return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos = _tensor_or_none(generation_config.bos_token_id, device)
        eos = _tensor_or_none(generation_config.eos_token_id, device)
        pad = _tensor_or_none(generation_config.pad_token_id, device)
        mask = _tensor_or_none(generation_config.mask_token_id, device)

        if eos is not None and eos.ndim == 0: eos = eos.unsqueeze(0)
        if pad is None and eos is not None: pad = eos[0]

        generation_config._bos_token_tensor = bos
        generation_config._eos_token_tensor = eos
        generation_config._pad_token_tensor = pad
        generation_config._mask_token_tensor = mask

    # [NEW FUNCTION] 复用之前定义的 Kick-start 初始化函数
    def initialize_with_kickstart(
        self,
        x: torch.LongTensor,
        prompt_len: int,
        kickstart_ids: Optional[torch.Tensor],
        kickstart_strength: float,
        kickstart_threshold: float,
        projection_type: str,
        mask_token_id: int,
        pad_token_id: Optional[int], # NEW argument
        attention_mask: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor],
        steps: int # 这里主要用于计算 fallback ratio
    ) -> Tuple[torch.LongTensor, int]: # Returns x and nfe_cost
        """
        RAG Kick-start 初始化：注入草稿并执行全局重掩码。
        在 Blockwise 模式下，我们仍然先进行全局初始化，然后在 Block 循环中处理局部 mask。
        """
        nfe_cost = 0
        if kickstart_ids is None or kickstart_strength >= 1.0:
            return x, nfe_cost

        gen_len = x.shape[1] - prompt_len

        # 1. 注入草稿
        draft_len = kickstart_ids.shape[1]
        valid_len = min(draft_len, gen_len)
        x[:, prompt_len : prompt_len + valid_len] = kickstart_ids[:, :valid_len]
        
        # [NEW] Mask Padding in Draft
        if pad_token_id is not None:
            padding_mask = (x[:, prompt_len:] == pad_token_id)
            x[:, prompt_len:][padding_mask] = mask_token_id
            
        is_draft = (x[:, prompt_len:] != mask_token_id)
        if pad_token_id is not None:
            is_draft = is_draft & (x[:, prompt_len:] != pad_token_id)

        remask_indices = None
        
        # 2. 投影与重掩码
        if projection_type == 'confidence' or kickstart_strength == -1:
            # 前向传播评估草稿质量
            outputs = self(x, attention_mask, tok_idx)
            nfe_cost += 1 # Count NFE
            logits = outputs.logits
            # Logit Shift
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            gen_logits = logits[:, prompt_len:]
            probs = torch.softmax(gen_logits, dim=-1)
            token_confidences = torch.gather(probs, -1, x[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
            token_confidences.masked_fill_(~is_draft, 100.0)
            
            if kickstart_strength == -1:
                remask_indices = (token_confidences < kickstart_threshold) & is_draft
            else:
                num_draft = is_draft.sum(dim=1)
                k = (num_draft.float() * kickstart_strength).long()
                remask_indices = torch.zeros_like(is_draft, dtype=torch.bool)
                for b in range(x.shape[0]):
                    if k[b] > 0:
                        _, topk_idx = torch.topk(token_confidences[b], k=k[b], largest=False)
                        remask_indices[b, topk_idx] = True
                        
        elif projection_type == 'random':
            probs = torch.rand_like(x[:, prompt_len:].float())
            remask_indices = (probs < kickstart_strength) & is_draft

        # 3. 应用重掩码
        if remask_indices is not None:
            x[:, prompt_len:][remask_indices] = mask_token_id
            
        return x, nfe_cost

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        # [NEW] Kick-start 参数
        kickstart_ids: Optional[torch.Tensor] = None,
        kickstart_strength: float = 1.0,
        kickstart_threshold: float = 0.9,
        projection_type: str = 'confidence',
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # Expand inputs
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        # Expand draft
        if kickstart_ids is not None:
             kickstart_ids, _ = self._expand_inputs_for_generation(
                expand_size=generation_config.num_return_sequences,
                input_ids=kickstart_ids
            )

        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", 32)
        dual_cache = kwargs.get("dual_cache", False)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            # [NEW] Pass args
            kickstart_ids=kickstart_ids,
            kickstart_strength=kickstart_strength,
            kickstart_threshold=kickstart_threshold,
            projection_type=projection_type
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
        # [NEW] Args
        kickstart_ids: Optional[torch.Tensor] = None,
        kickstart_strength: float = 1.0,
        kickstart_threshold: float = 0.9,
        projection_type: str = 'confidence',
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        
        # Init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        pad_token_id = generation_config.pad_token_id # Get pad token
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        eps = generation_config.eps

        nfe = 0
        histories = [] if (return_dict_in_generate and output_history) else None

        # 1. Initialize Canvas
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        prompt_len = input_ids.shape[1]
        gen_length = max_length - prompt_len
        
        # 2. Prepare Attention Mask (Global)
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # 3. [NEW] Kick-start Global Initialization
        # 这会填充 x 并根据置信度 mask 掉不确定的部分
        x, init_nfe = self.initialize_with_kickstart(
            x=x,
            prompt_len=prompt_len,
            kickstart_ids=kickstart_ids,
            kickstart_strength=kickstart_strength,
            kickstart_threshold=kickstart_threshold,
            projection_type=projection_type,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            attention_mask=attention_mask,
            tok_idx=tok_idx,
            steps=steps
        )
        nfe += init_nfe # ADD nfe count

        # Handle block configuration
        if block_length is None:
            block_length = gen_length 
        
        assert gen_length % block_length == 0, f"gen_length {gen_length} / block_length {block_length}"
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0, f"steps {steps} / num_blocks {num_blocks}"
        
        # 基础步数（用于全 Mask 情况）
        base_steps_per_block = steps // num_blocks
        
        past_key_values = None

        # Process each block
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # [MODIFIED] 计算当前 Block 的初始 Mask 比例
            # 用于决定该 Block 需要多少步，以及 Time Travel 的起点
            block_mask_count = (x[:, current_block_start:current_block_end] == mask_token_id).sum().item()
            total_block_tokens = x.shape[0] * block_length
            block_mask_ratio = block_mask_count / total_block_tokens
            
            # 边界处理
            if block_mask_ratio < 1e-4: block_mask_ratio = 1.0 / base_steps_per_block

            # 计算该 Block 的有效步数
            effective_steps_per_block = int(base_steps_per_block * block_mask_ratio)
            if effective_steps_per_block < 1: effective_steps_per_block = 1

            # [MODIFIED] 生成当前 Block 的时间步
            # 动态调整起点：从 block_mask_ratio 开始
            timesteps = torch.linspace(block_mask_ratio, eps, effective_steps_per_block + 1, device=x.device)

            # --- Pre-computation / Caching Step ---
            # 即使 Block 已经填满，也需要跑一次 Forward 更新 KV Cache，为下一个 Block 做准备
            
            model_output = self(x, attention_mask, tok_idx, use_cache=True)
            nfe += 1
            
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1) # Logit Shift
            
            # [MODIFIED] 仅当起始位置是 Mask 时，才尝试用 sampling 更新
            # 如果 Kick-start 已经填了值，我们信任 Kick-start（经过 Projection 筛选后的）
            is_masked_start = (x[:, current_block_start] == mask_token_id)
            if is_masked_start.any():
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                # 仅更新 Mask 的部分
                x[:, current_block_start][is_masked_start] = x0[:, current_block_start][is_masked_start]

            # Cache management
            if not dual_cache:
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
                past_key_values = new_past_key_values
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1
                
            # --- Diffusion Loop (Blockwise) ---
            
            i = 0 # 计数器归零
            # [MODIFIED] 循环次数改为 effective_steps_per_block
            # 这样可以跳过那些草稿已经填充好的步骤
            while i < effective_steps_per_block:
                
                # Check Mask status
                if dual_cache:
                    mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                else:
                    mask_index = (x[:, current_block_start:] == mask_token_id)
                
                # 提前退出：如果该 Block 已经没有 Mask 了
                if not mask_index.any():
                    break

                # Prepare attention mask
                if attention_mask != "full":
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask
                
                # Forward with Cache
                if dual_cache:
                    model_output = self(x[:, current_block_start:current_block_end], current_attention_mask, 
                                    tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True, dual_cache=dual_cache, replace_position=replace_position)
                else:
                    model_output = self(x[:, current_block_start:], current_attention_mask, 
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                nfe += 1
                
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1) # Logit Shift

                # === Algorithm Selection ===
                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    if dual_cache:
                        x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_index] = x0.clone()
                    full_confidence[mask_index] = confidence
                    # 保护未来 block 不被选中（Standard Cache 模式下 logits 会包含未来 block）
                    if not dual_cache:
                        full_confidence[:, block_length:] = -torch.inf
                    
                    # 计算需要 transfer 的数量
                    current_mask_num = mask_index.sum().item()
                    # 动态分配每步配额
                    tokens_to_transfer = max(1, current_mask_num // (effective_steps_per_block - i))
                    
                    selected_confidence, select_index = torch.topk(full_confidence, tokens_to_transfer)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    # 简化处理 batch=1
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    
                    for k in range(tokens_to_transfer):
                        if selected_confidence[0, k] < threshold and i < effective_steps_per_block - 1:
                            transfer_index[0, select_index[0, k]] = False

                    if dual_cache:
                        x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                    else:
                        x[:, current_block_start:][transfer_index] = x_[transfer_index]

                else:
                    # Origin / MaskGit+ / etc.
                    # 使用动态调整后的 t 和 s
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    
                    if not dual_cache:
                        mask_index[:, block_length:] = False
                    
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < effective_steps_per_block - 1 else int(num_mask_token)
                    
                    if dual_cache:
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    full_confidence[mask_index] = confidence
                    if not dual_cache:
                        full_confidence[:, block_length:] = -torch.inf
                    
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        
                        if dual_cache:
                            x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                            
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        
                        if dual_cache:
                            x[:, current_block_start:current_block_end][row_indices,transfer_index] = x_[row_indices,transfer_index]
                        else:
                            x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                
                i += 1

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe=nfe, 
            )
        else:
            return x