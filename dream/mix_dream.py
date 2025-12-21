import os
import sys
import time
import gc
import argparse
import random
import torch
import types
import numpy as np
from typing import Optional, Tuple

# 假设运行目录在 dream/ 下，添加上级目录以导入 lm_evals 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
from model.modeling_dream import DreamModel

# 导入 RAG 和评估相关模块 (来自 lm_evals)
try:
    from lm_evals.dataset_processor import (
        process_prompt,
        process_retrieved_kickstart_text,
        split_tarin_test,
        load_evaluation_metrics
    )
    from lm_evals.evaluation import compute_metrics
    from lm_evals.rag_engine import AdvancedRAGEngine
    from lm_evals.rag2 import FlashRAGEngine
except ImportError:
    # 如果路径不对，尝试直接导入 (假设已在 PYTHONPATH 中)
    from dataset_processor import (
        process_prompt,
        process_retrieved_kickstart_text,
        split_tarin_test,
        load_evaluation_metrics
    )
    from evaluation import compute_metrics
    from rag_engine import AdvancedRAGEngine


# ==================== 常量定义 ====================
MASK_ID = 126336  # Dream 的 [MASK] token ID

def set_seed(seed: int = 42):
    """设置随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Dream RAG Kick-start & Fast-dLLM 融合评估脚本")
    
    # --- 模型与基础配置 ---
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/model/Dream-v0-Instruct-7B', help='Dream 模型路径')
    parser.add_argument('--dataset', type=str, default='arc', help='数据集: mmlu, arc, gsm8k 等')
    parser.add_argument('--data_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--num_test_examples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小 (Dual Cache模式下建议为1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_result', action='store_true', help='保存详细结果')
    parser.add_argument('--verbose', action='store_true', help='打印详细日志')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='如0.1表示只用10%数据')

    # --- 生成参数 (Dream & Fast-dLLM) ---
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--steps', type=int, default=128, help='扩散总步数')
    parser.add_argument('--block_length', type=int, default=32, help='Block 长度')
    parser.add_argument('--temperature', type=float, default=0., help='采样温度')
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--use_cache', action='store_true', help='启用 KV Cache')
    parser.add_argument('--dual_cache', action='store_true', help='启用 Dual Cache (Fast-dLLM关键特性)')
    parser.add_argument('--threshold', type=float, default=0.9, help='置信度阈值 (用于 confidence_threshold 算法)')
    parser.add_argument('--alg', type=str, default="entropy", help='采样算法: origin, confidence_threshold等')

    # --- RAG 与 Kick-start 参数 ---
    parser.add_argument('--rag_engine', type=str, default='ours', choices=['ours', 'flashrag'], help='RAG 引擎')
    parser.add_argument('--embed_model', type=str, default='BAAI/bge-small-zh-v1.5', help='Embedding 模型路径')
    parser.add_argument('--rerank_model', type=str, default='BAAI/bge-reranker-v2-m3', help='Rerank 模型路径')
    parser.add_argument('--test_top_k', type=int, default=1, help='测试检索 Top-K 的深度')
    parser.add_argument('--kickstart_strength', type=float, default=None, help='指定单一强度 (0.0-1.0)。若不指定则运行网格搜索。')
    parser.add_argument('--projection_type', type=str, default='confidence', choices=['confidence', 'random'], help='草稿投影策略')
    parser.add_argument('--option_padding', type=int, default=0, help='选择题选项前的强制 Mask 填充数')

    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 打印配置头
    print("\n" + "=" * 80)
    print(f"{'MIX_DREAM: RAG Kick-start + Fast-dLLM':^80}")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"数据: {args.dataset.upper()} | 样本数: {args.num_test_examples}")
    print(f"生成: Steps={args.steps}, Block={args.block_length}, DualCache={args.dual_cache}")
    print(f"RAG : Engine={args.rag_engine}, TopK={args.test_top_k}")
    print("=" * 80 + "\n")

    # 1. 加载模型与分词器
    print(">>> Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left' # 生成任务通常使用左填充

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    # 2. 动态 Patch 模型方法 (关键步骤)
    # 根据是否开启 Cache 选择不同的 Generation Mixin
    print(f">>> Patching Model Methods (use_cache={args.use_cache})...")
    if args.use_cache:
        # 使用 generation_utils_block.py (支持 Blockwise, Dual Cache)
        # 假设文件已更新为支持 Kick-start 的版本
        from model.generation_utils_block import DreamGenerationMixin
    else:
        # 使用 generation_utils.py (标准扩散)
        # 假设文件已更新为支持 Kick-start 的版本
        from model.generation_utils import DreamGenerationMixin
    
    # 覆盖原模型的方法
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    # 确保 initialize_with_kickstart 也被绑定 (如果在 Mixin 中定义了)
    if hasattr(DreamGenerationMixin, 'initialize_with_kickstart'):
         model.initialize_with_kickstart = types.MethodType(DreamGenerationMixin.initialize_with_kickstart, model)

    # 3. 准备数据 [关键修复: 传入 data_ratio]
    print(f">>> Loading Dataset: {args.dataset} (Ratio={args.data_ratio})...")
    train_prompts, train_targets, test_prompts, test_targets = split_tarin_test(
        args.dataset, args.data_path, data_ratio=args.data_ratio
    )
    print(f"    Train Size (KB): {len(train_prompts)}")
    print(f"    Test Size: {len(test_prompts)}")

    # 截断测试集
    NUM_TEST_EXAMPLES = min(args.num_test_examples, len(test_prompts))
    test_prompts = test_prompts[:NUM_TEST_EXAMPLES]
    test_targets = test_targets[:NUM_TEST_EXAMPLES]

    # 4. 构建 RAG 知识库
    print(">>> Building RAG Knowledge Base...")
    if args.rag_engine == 'flashrag':
        from rag2 import FlashRAGEngine
        rag = FlashRAGEngine()
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    else:
        rag = AdvancedRAGEngine(embed_model=args.embed_model, rerank_model=args.rerank_model, device=device)
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    print("    RAG Ready.")

    # 5. 定义实验网格 (Strength x Rank)
    if args.kickstart_strength is not None:
        strengths_to_test = [args.kickstart_strength]
    else:
        # 默认网格: 包含 自适应(-1), 纯RAG(0.0), 以及不同比例的混合
        # 注意: 0.0 在 mix.py 逻辑中通常指直接输出草稿，但在 Dream 代码中 kickstart_strength=0.0 意味着全信草稿
        # 为了与 Dream 逻辑一致: 0.0 (全信草稿), 1.0 (全重绘/不使用草稿)
        # 这里的 steps 用于生成比例列表
        base_steps = [0, 32, 64, 96, 128] # 示例步数点
        strengths_to_test = [float(s/128) for s in base_steps]
        strengths_to_test.insert(0, -1) # 添加自适应模式
        # 确保 1.0 在列表最后作为 Baseline
        if 1.0 not in strengths_to_test: strengths_to_test.append(1.0)
    
    ranks_to_test = list(range(args.test_top_k))

    # 初始化统计容器
    stats_storage = {}
    for s in strengths_to_test:
        for r in ranks_to_test:
            stats_storage[(s, r)] = {
                'total_time': 0, 'total_nfe': 0, 'predictions': [], 'references': []
            }

    # 6. 批量推理循环
    batch_size = args.batch_size
    # Dual Cache 强制 batch=1
    if args.dual_cache and batch_size > 1:
        print("[Warning] Dual Cache enabled, forcing batch_size=1")
        batch_size = 1

    num_batches = (NUM_TEST_EXAMPLES + batch_size - 1) // batch_size
    
    print(f"\n>>> Starting Inference Loop ({num_batches} batches)...")
    
    # 预热 (使用 Baseline 配置)
    print("    [Warmup] Running one generation...")
    warmup_input = tokenizer(["Hello"], return_tensors='pt').to(device)
    try:
        model.diffusion_generate(
            warmup_input['input_ids'], attention_mask=warmup_input['attention_mask'],
            steps=min(16, args.steps), block_length=args.block_length,
            dual_cache=args.dual_cache, use_cache=args.use_cache
        )
    except Exception as e:
        print(f"    [Warmup Failed] {e}")

    rag_search_time = 0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, NUM_TEST_EXAMPLES)
        current_bs = batch_end - batch_start
        
        batch_prompts_raw = test_prompts[batch_start:batch_end]
        batch_targets_raw = test_targets[batch_start:batch_end]
        
        print(f"    Processing Batch {batch_idx+1}/{num_batches}...")

        # A. RAG 检索
        t0 = time.perf_counter()
        # 检索 Top-K 个结果
        retrieval_results, _ = rag.batch_search(
            batch_prompts_raw, 
            top_k_retrieve=args.test_top_k, 
            top_n_rerank=args.test_top_k,
            batch_size=current_bs
        )
        rag_search_time += (time.perf_counter() - t0)

        # B. 准备模型输入
        # 应用 Chat Template
        batch_formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": process_prompt(p, args.dataset)}],
                add_generation_prompt=True, tokenize=False
            ) for p in batch_prompts_raw
        ]
        inputs = tokenizer(batch_formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        # 缓存 Baseline (Strength=1.0) 的结果，避免每个 Rank 重复跑
        baseline_results = None

        # C. 实验网格循环
        for strength in strengths_to_test:
            for rank in ranks_to_test:
                
                # 优化: 如果是 Baseline (1.0)，且已经跑过，直接复用
                if strength == 1.0 and baseline_results is not None:
                    stats_storage[(strength, rank)]['total_time'] += baseline_results['time']
                    stats_storage[(strength, rank)]['total_nfe'] += baseline_results['nfe']
                    stats_storage[(strength, rank)]['predictions'].extend(baseline_results['preds'])
                    stats_storage[(strength, rank)]['references'].extend(batch_targets_raw)
                    continue

                # 1. 准备 Kick-start IDs (草稿)
                kickstart_texts = []
                for res in retrieval_results:
                    if res and len(res) > rank:
                        draft = res[rank]['target']
                    else:
                        draft = "" # 无检索结果
                    kickstart_texts.append(process_retrieved_kickstart_text(draft, args.dataset))
                
                # Tokenize 草稿
                kickstart_inputs = tokenizer(kickstart_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                kickstart_ids = kickstart_inputs['input_ids']

                # 选择题特殊处理 (Option Padding)
                if args.option_padding > 0:
                    pad_tensor = torch.full((current_bs, args.option_padding), MASK_ID, device=device, dtype=torch.long)
                    kickstart_ids = torch.cat([pad_tensor, kickstart_ids], dim=1)

                # 2. 执行生成
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                
                # 调用 diffusion_generate (会自动调用 initialize_with_kickstart)
                output = model.diffusion_generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=args.gen_length,
                    return_dict_in_generate=True,
                    # 基础参数
                    steps=args.steps,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    threshold=args.threshold,
                    alg=args.alg,
                    # Fast-dLLM 参数
                    dual_cache=args.dual_cache,
                    use_cache=args.use_cache,
                    # Kick-start 参数
                    kickstart_ids=kickstart_ids,
                    kickstart_strength=strength,
                    projection_type=args.projection_type
                )
                
                torch.cuda.synchronize()
                t_cost = time.perf_counter() - t_gen_start
                
                # 3. 解码与存储
                prompt_len = inputs['input_ids'].shape[1]
                preds = [
                    tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    for seq in output.sequences
                ]
                
                nfe = getattr(output, 'nfe', 0) # 获取 NFE

                stats_storage[(strength, rank)]['total_time'] += t_cost
                stats_storage[(strength, rank)]['total_nfe'] += nfe
                stats_storage[(strength, rank)]['predictions'].extend(preds)
                stats_storage[(strength, rank)]['references'].extend(batch_targets_raw)
                
                # 缓存 Baseline
                if strength == 1.0:
                    baseline_results = {'time': t_cost, 'nfe': nfe, 'preds': preds}

                if args.verbose and batch_idx == 0:
                    print(f"      [Str={strength}, Rank={rank}] NFE={nfe}, Time={t_cost:.2f}s")

    # 7. 评估与输出表格
    print("\n" + "=" * 80)
    print(f"{'EVALUATION RESULTS':^80}")
    print("=" * 80)
    
    metric_objs = load_evaluation_metrics(args.dataset)
    
    # 打印表头
    headers = ["Strength", "Rank", "Time(s)", "Avg NFE"] 
    # 动态获取 Metric 名字 (跑一次空的 compute_metrics 获取 key)
    # 这里直接假设 compute_metrics 返回字典
    sample_metrics = compute_metrics([""], [""], args.dataset, metric_objs)
    metric_keys = list(sample_metrics.keys())
    headers.extend(metric_keys)
    
    row_format = "{:<10} {:<6} {:<10} {:<10}" + " {:<12}" * len(metric_keys)
    print(row_format.format(*headers))
    print("-" * 80)

    # 排序输出: 优先按 Rank, 然后按 Strength (从自适应到1.0)
    sorted_keys = sorted(stats_storage.keys(), key=lambda x: (x[1], x[0]))
    
    for s, r in sorted_keys:
        data = stats_storage[(s, r)]
        preds = data['predictions']
        refs = data['references']
        
        # 计算指标
        scores = compute_metrics(preds, refs, args.dataset, metric_objs)
        
        # 准备行数据
        avg_time = data['total_time'] / NUM_TEST_EXAMPLES
        avg_nfe = data['total_nfe'] / NUM_TEST_EXAMPLES if NUM_TEST_EXAMPLES > 0 else 0
        
        str_val = "Adaptive" if s == -1 else f"{s:.2f}"
        row_vals = [str_val, r, f"{avg_time:.3f}", f"{avg_nfe:.1f}"]
        for k in metric_keys:
            row_vals.append(f"{scores.get(k, 0):.4f}")
            
        print(row_format.format(*row_vals))
        
        # 保存到文件
        if args.save_result:
            os.makedirs("results_dream", exist_ok=True)
            fname = f"results_dream/{args.dataset}_str{s}_rank{r}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                for p in preds:
                    f.write(p.replace('\n', ' ') + '\n')

    print("-" * 80)
    print(f"RAG Total Search Time: {rag_search_time:.2f}s")
    print("Done.")

if __name__ == "__main__":
    main()