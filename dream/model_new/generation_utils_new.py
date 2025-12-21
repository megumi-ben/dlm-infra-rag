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

import time
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
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    # [NEW FUNCTION] 封装热启动初始化逻辑
    def initialize_with_kickstart(
        self,
        x: torch.LongTensor,
        prompt_len: int,
        kickstart_ids: Optional[torch.Tensor],
        kickstart_strength: float,
        kickstart_threshold: float,
        projection_type: str,
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor],
        steps: int
    ) -> Tuple[torch.LongTensor, float]:
        """
        处理 RAG 热启动逻辑：注入草稿、评估置信度、执行重掩码。
        返回初始化后的画布 x 和 初始 Mask 比例。
        """
        # 默认初始状态：全 Mask，比例为 1.0
        initial_mask_ratio = 1.0
        
        # 如果没有草稿或强度为 1.0（全重绘），直接返回全 Mask 的 x
        if kickstart_ids is None or kickstart_strength >= 1.0:
            return x, initial_mask_ratio

        gen_len = x.shape[1] - prompt_len

        # 1. 注入草稿 (Draft Injection)
        draft_len = kickstart_ids.shape[1]
        valid_len = min(draft_len, gen_len)
        # 将草稿填入生成区
        x[:, prompt_len : prompt_len + valid_len] = kickstart_ids[:, :valid_len]
        
        # 标记真实草稿位置 (非 Mask 且 非 Padding)
        is_draft = (x[:, prompt_len:] != mask_token_id)
        
        remask_indices = None
        
        # 2. 投影与重掩码 (Projection & Re-masking)
        if projection_type == 'confidence' or kickstart_strength == -1:
            # === Confidence Mode ===
            # 前向传播评估草稿质量
            outputs = self(x, attention_mask, tok_idx)
            logits = outputs.logits
            
            # [CRITICAL] Dream Logit Shift 修正
            # 必须执行这一步，否则 Logits 和 Token 位置错位
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            gen_logits = logits[:, prompt_len:]
            probs = torch.softmax(gen_logits, dim=-1)
            
            # 获取草稿 Token 的置信度
            # x 维度 [B, L], unsqueeze 后 [B, L, 1] 用于 gather
            token_confidences = torch.gather(probs, -1, x[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
            
            # 保护非草稿区域（本来就是 Mask）不被选中
            token_confidences.masked_fill_(~is_draft, 100.0)
            
            if kickstart_strength == -1:
                # 自适应阈值模式
                remask_indices = (token_confidences < kickstart_threshold) & is_draft
            else:
                # 固定比例模式：Mask 掉置信度最低的 N 个
                num_draft = is_draft.sum(dim=1)
                k = (num_draft.float() * kickstart_strength).long()
                
                remask_indices = torch.zeros_like(is_draft, dtype=torch.bool)
                for b in range(x.shape[0]):
                    if k[b] > 0:
                        # 找出最低置信度的 k 个位置
                        _, topk_idx = torch.topk(token_confidences[b], k=k[b], largest=False)
                        remask_indices[b, topk_idx] = True
                        
        elif projection_type == 'random':
            # === Random Mode ===
            probs = torch.rand_like(x[:, prompt_len:].float())
            remask_indices = (probs < kickstart_strength) & is_draft

        # 3. 应用重掩码
        if remask_indices is not None:
            x[:, prompt_len:][remask_indices] = mask_token_id
        
        # 4. 计算剩余 Mask 比例
        # 用于后续的时间步调度 (Time Travel)
        current_mask_count = (x[:, prompt_len:] == mask_token_id).sum().item()
        total_gen_tokens = x.shape[0] * gen_len
        if total_gen_tokens > 0:
            initial_mask_ratio = current_mask_count / total_gen_tokens
        else:
            initial_mask_ratio = 1.0 # Fallback
            
        # 边界保护：防止除零或过小导致步数为0
        if initial_mask_ratio < 1e-4: 
            initial_mask_ratio = 1.0 / steps

        return x, initial_mask_ratio
    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        # [NEW] RAG Kick-start 参数
        kickstart_ids: Optional[torch.Tensor] = None,
        kickstart_strength: float = 1.0,      # 1.0=不使用草稿(全Mask), 0.0=全信草稿
        kickstart_threshold: float = 0.9,     # 用于 confidence 模式的筛选阈值
        projection_type: str = 'confidence',  # 'confidence' 或 'random'
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle generation_config
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare max_length
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check device (omitted warnings for brevity, same as original)

        # 5. Expand inputs
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        # [NEW] 扩展草稿维度以匹配 batch_size
        if kickstart_ids is not None:
            kickstart_ids, _ = self._expand_inputs_for_generation(
                expand_size=generation_config.num_return_sequences,
                input_ids=kickstart_ids
            )

        threshold = kwargs.get("threshold", 0.9)

        # 6. Call sample
        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            threshold=threshold,
            # [NEW] 传递 Kick-start 参数
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
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        # [NEW] 接收 Kick-start 参数
        kickstart_ids: Optional[torch.Tensor] = None,
        kickstart_strength: float = 1.0,
        kickstart_threshold: float = 0.9,
        projection_type: str = 'confidence',
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        start_time = time.time()
        
        # [MODIFIED] 画布初始化逻辑
        prompt_len = input_ids.shape[1]
        gen_len = max_length - prompt_len
        
        # 先创建一个全 Mask 的画布
        x = F.pad(input_ids, (0, gen_len), value=mask_token_id)
        
        # [MODIFIED] Attention Mask 处理前置
        # 因为 initialize_with_kickstart 可能需要进行 forward 计算置信度，
        # 所以必须先构建好 attention_mask 和 tok_idx
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

        # [NEW] 调用 Kick-start 初始化
        x, initial_mask_ratio = self.initialize_with_kickstart(
            x=x,
            prompt_len=prompt_len,
            kickstart_ids=kickstart_ids,
            kickstart_strength=kickstart_strength,
            kickstart_threshold=kickstart_threshold,
            projection_type=projection_type,
            mask_token_id=mask_token_id,
            attention_mask=attention_mask,
            tok_idx=tok_idx,
            steps=steps
        )

        # [MODIFIED] 自适应调度 (Adaptive Scheduling)
        # 根据 Mask 比例动态计算 effective_steps，实现加速
        effective_steps = int(steps * initial_mask_ratio)
        if effective_steps < 1: effective_steps = 1
        
        # 时间步重映射：从 initial_mask_ratio 开始，而不是从 1.0 开始
        # 这样 s/t 的计算会自动适配缩短后的进程
        timesteps = torch.linspace(initial_mask_ratio, eps, effective_steps + 1, device=x.device)

        # hook
        x = generation_tokens_hook_func(None, x, None)
        
        # 针对 confidence_threshold 算法的变量准备
        if alg == 'confidence_threshold':
            mask_index = (x == mask_token_id)
            total_mask_num = mask_index.sum().item()
            
            # [MODIFIED] 适配 effective_steps
            # 重新计算每步的配额。如果步数变少了，每步步子要迈大一点
            if effective_steps > 0:
                number_transfer_tokens = total_mask_num // effective_steps
            else:
                number_transfer_tokens = total_mask_num
            left_tokens_last_step = 0
            
            # 为了兼容原代码逻辑，这里不做严格的 batch size 校验，但逻辑本身通常假设 batch=1
        
        i = 0
        # [MODIFIED] 循环次数改为 effective_steps
        while i < effective_steps:
            mask_index = (x == mask_token_id)
            # 提前退出检查
            if not mask_index.any():
                break
                
            logits = self(x, attention_mask, tok_idx).logits
            # [CRITICAL] Dream Logit Shift
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            logits = generation_logits_hook_func(i, x, logits)
            mask_logits = logits[mask_index]
            
            if not alg == 'confidence_threshold':
                t = timesteps[i]
                s = timesteps[i + 1]
            
            # === 算法分支 ===
            if alg == 'origin':
                # i 是从 0 到 effective_steps-1
                p_transfer = 1 - s / t if i < effective_steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            
            elif alg == 'confidence_threshold':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                current_transfer_tokens = number_transfer_tokens + left_tokens_last_step
                left_tokens_last_step = 0
                
                # 防止请求的数量超过实际剩余 Mask 数
                valid_mask_count = mask_index.sum().item()
                if current_transfer_tokens > valid_mask_count:
                    current_transfer_tokens = valid_mask_count
                
                selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                select_index = select_index.to(x.device)
                
                # 注意：原代码此处隐含假设 batch_size=1
                transfer_index[0, select_index[0]] = True
                
                # 阈值过滤逻辑
                # 遍历所有选中的 token，如果不满足阈值且不是最后一步，则推迟到下一步
                for k in range(current_transfer_tokens): 
                     if selected_confidence[0, k] < threshold:
                        if i < effective_steps - 1:
                            left_tokens_last_step += 1
                            transfer_index[0, select_index[0, k]] = False
                        else:
                            # 最后一步，强制接受，或者选择不生成（取决于策略）
                            # 为了保证生成完成，这里选择不再推迟
                            pass 

                x[transfer_index] = x_[transfer_index].clone()

            else:
                # maskgit_plus, topk_margin, entropy
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                # [MODIFIED] 使用适配后的 t 和 s 计算 transfer 数量
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < effective_steps - 1 else int(num_mask_token)
                
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
            i += 1
        
        print(f'used steps: {effective_steps} (Original plan: {steps})')
        end_time = time.time()
        print(f'used time: {end_time - start_time}')
        
        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories)
        else:
            return x