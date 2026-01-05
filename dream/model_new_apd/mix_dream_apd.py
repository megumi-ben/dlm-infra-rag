import os
import sys
import time
import argparse
import random
import torch
import types
import numpy as np
from typing import Optional, Tuple

# 假设运行目录在 dream/ 下，添加上级目录以导入 lm_evals 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入 Dream 模型
try:
    from model_apd.modeling_dream import DreamModel
    from model_apd.generation_utils import DreamGenerationMixin
except ImportError:
    print("Warning: Could not import from model_apd directly, checking path...")
    from dream.model_apd.modeling_dream import DreamModel
    from dream.model_apd.generation_utils import DreamGenerationMixin

# 导入 RAG 和评估模块
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
    # 本地调试时的 fallback
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
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Dream APD (RAG Kick-start) 评估脚本")
    
    # --- 模型路径 ---
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/model/Dream-v0-Instruct-7B', help='Dream Diffusion 模型路径')
    parser.add_argument('--verifier_path', type=str, default='/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct', help='[APD必须] Verifier 辅助模型路径')
    
    # --- 数据配置 ---
    parser.add_argument('--dataset', type=str, default='arc', help='数据集: mmlu, arc, gsm8k 等')
    parser.add_argument('--data_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--num_test_examples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小 (APD 模式强烈建议为 1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_result', action='store_true', help='保存结果')
    parser.add_argument('--verbose', action='store_true', help='打印详细日志')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='数据比例')
    parser.add_argument('--dynamic_gen_length', action='store_true', help='动态调整生成长度')

    # --- 生成参数 ---
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--steps', type=int, default=512, help='扩散总步数 (APD 中一般用于调度)')
    parser.add_argument('--temperature', type=float, default=0.2, help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=None)
    
    # --- APD 核心参数 ---
    parser.add_argument("--alg", type=str, default="apd", choices=["apd", "origin", "leftright", "entropy", "low_confidence"], help="解码算法")
    parser.add_argument('--apd_mixture_weight', type=float, default=0.7, help='[APD] R值: 混合权重 (0.0-1.0)')
    parser.add_argument('--kv_window', type=int, default=16, help='[APD] W值: KV Cache 窗口大小')
    parser.add_argument('--max_lookahead', type=int, default=100, help='[APD] M值: 最大并行前瞻步数')
    
    # --- RAG & Kick-start 参数 ---
    parser.add_argument('--rag_engine', type=str, default='ours', choices=['ours', 'flashrag'], help='RAG 引擎')
    parser.add_argument('--embed_model', type=str, default='BAAI/bge-small-zh-v1.5', help='Embedding 模型')
    parser.add_argument('--rerank_model', type=str, default='BAAI/bge-reranker-v2-m3', help='Rerank 模型')
    parser.add_argument('--test_top_k', type=int, default=1, help='测试检索 Top-K')
    parser.add_argument('--kickstart_strength', type=float, default=None, help='指定单一强度。不指定则运行网格搜索。')
    parser.add_argument('--projection_type', type=str, default='confidence', choices=['confidence', 'random'], help='草稿投影策略')
    parser.add_argument('--option_padding', type=int, default=0, help='选择题强制占位符')

    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 打印配置
    print("\n" + "=" * 80)
    print(f"{'MIX_DREAM_APD: RAG Kick-start + APD':^80}")
    print("=" * 80)
    print(f"Dataset   : {args.dataset.upper()} (N={args.num_test_examples})")
    print(f"Diffusion : {args.model_path}")
    print(f"Verifier  : {args.verifier_path}")
    print(f"APD Config: Alg={args.alg}, R={args.apd_mixture_weight}, W={args.kv_window}, M={args.max_lookahead}")
    print(f"RAG Config: Engine={args.rag_engine}, TopK={args.test_top_k}")
    print("=" * 80 + "\n")

    # 1. 加载模型与分词器
    print(">>> Loading Models...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # Diffusion Model
    print("    [1/2] Loading Diffusion Model...")
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="sdpa", # 如果 Dream 支持 sdpa 可以开启
        device_map=device
    ).eval()

    # Verifier Model
    print(f"    [2/2] Loading Verifier Model ({args.verifier_path})...")
    verifier_model = AutoModelForCausalLM.from_pretrained(
        args.verifier_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa", # 显式开启 SDPA 加速
        device_map=device
    ).eval()

    # 2. Patch 模型方法 (注入 APD 和 Kickstart)
    print(f">>> Patching Model Methods...")
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model.apd_sample = types.MethodType(DreamGenerationMixin.apd_sample, model)
    
    if hasattr(DreamGenerationMixin, 'initialize_with_kickstart'):
         model.initialize_with_kickstart = types.MethodType(DreamGenerationMixin.initialize_with_kickstart, model)
    else:
        raise ImportError("initialize_with_kickstart not found in DreamGenerationMixin. Please update generation_utils.py.")

    # 3. 准备数据
    print(f">>> Loading Dataset: {args.dataset}...")
    train_prompts, train_targets, test_prompts, test_targets = split_tarin_test(
        args.dataset, args.data_path, data_ratio=args.data_ratio
    )
    print(f"    Train: {len(train_prompts)}, Test: {len(test_prompts)}")

    # 截断测试集
    NUM_TEST_EXAMPLES = min(args.num_test_examples, len(test_prompts))
    test_prompts = test_prompts[:NUM_TEST_EXAMPLES]
    test_targets = test_targets[:NUM_TEST_EXAMPLES]

    # 4. 构建 RAG
    print(">>> Building RAG Knowledge Base...")
    if args.rag_engine == 'flashrag':
        from rag2 import FlashRAGEngine
        rag = FlashRAGEngine()
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    else:
        rag = AdvancedRAGEngine(embed_model=args.embed_model, rerank_model=args.rerank_model, device=device)
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
    print("    RAG Ready.")

    # 5. 实验网格配置 (Strength x Rank)
    if args.kickstart_strength is not None:
        strengths_to_test = [args.kickstart_strength]
    else:
        # 默认对比: 自适应(-1), 纯RAG(0.0), 混合(0.5), Baseline(1.0)
        strengths_to_test = [-1, 0.0, 0.5, 1.0]
    
    ranks_to_test = list(range(args.test_top_k))

    stats_storage = {}
    for s in strengths_to_test:
        for r in ranks_to_test:
            stats_storage[(s, r)] = {
                'total_time': 0, 'total_nfe': 0, 'predictions': [], 'references': []
            }

    # 6. 推理循环
    batch_size = args.batch_size
    if batch_size > 1:
        print("[Warning] APD logic usually requires batch_size=1. Forcing batch_size=1.")
        batch_size = 1

    num_batches = (NUM_TEST_EXAMPLES + batch_size - 1) // batch_size
    
    print(f"\n>>> Starting Inference ({num_batches} batches)...")
    
    rag_search_time = 0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, NUM_TEST_EXAMPLES)
        current_bs = batch_end - batch_start
        
        batch_prompts_raw = test_prompts[batch_start:batch_end]
        batch_targets_raw = test_targets[batch_start:batch_end]
        
        print(f"    Batch {batch_idx+1}/{num_batches}...")

        # A. RAG 检索
        t0 = time.perf_counter()
        retrieval_results, _ = rag.batch_search(
            batch_prompts_raw, 
            top_k_retrieve=args.test_top_k, 
            top_n_rerank=args.test_top_k,
            batch_size=current_bs
        )
        rag_search_time += (time.perf_counter() - t0)

        # B. 模板处理
        batch_formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": process_prompt(p, args.dataset)}],
                add_generation_prompt=True, tokenize=False
            ) for p in batch_prompts_raw
        ]
        inputs = tokenizer(batch_formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        baseline_results = None

        # C. 网格搜索
        for strength in strengths_to_test:
            for rank in ranks_to_test:
                
                # Baseline 缓存优化 (1.0 = No Kickstart)
                if strength == 1.0 and baseline_results is not None:
                    stats_storage[(strength, rank)]['total_time'] += baseline_results['time']
                    stats_storage[(strength, rank)]['total_nfe'] += baseline_results['nfe']
                    stats_storage[(strength, rank)]['predictions'].extend(baseline_results['preds'])
                    stats_storage[(strength, rank)]['references'].extend(batch_targets_raw)
                    continue

                # 1. 准备 Draft
                kickstart_texts = []
                for res in retrieval_results:
                    if res and len(res) > rank:
                        draft = res[rank]['target']
                    else:
                        draft = ""
                    kickstart_texts.append(process_retrieved_kickstart_text(draft, args.dataset))
                
                kickstart_inputs = tokenizer(kickstart_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                kickstart_ids = kickstart_inputs['input_ids']

                # Option Padding (选择题)
                if args.option_padding > 0:
                    pad_tensor = torch.full((current_bs, args.option_padding), MASK_ID, device=device, dtype=torch.long)
                    kickstart_ids = torch.cat([pad_tensor, kickstart_ids], dim=1)

                # 动态长度
                if args.dynamic_gen_length:
                    real_gen_length = max(64, min(kickstart_ids.shape[1] + 64, 512)) # 稍微放宽一点
                else:
                    real_gen_length = args.gen_length

                # 2. 生成 (APD)
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                
                try:
                    output = model.diffusion_generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=real_gen_length,
                        return_dict_in_generate=True,
                        
                        # --- 核心 APD 参数 ---
                        alg=args.alg, # 默认为 apd
                        verifier_model=verifier_model, # [关键修正]
                        apd_mixture_weight=args.apd_mixture_weight,
                        kv_window=args.kv_window,
                        max_lookahead=args.max_lookahead,
                        
                        # --- 基础参数 ---
                        steps=args.steps,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        
                        # --- Kick-start 参数 ---
                        kickstart_ids=kickstart_ids,
                        kickstart_strength=strength,
                        kickstart_threshold=0.9, 
                        projection_type=args.projection_type
                    )
                    
                    if hasattr(output, 'sequences'):
                        seqs = output.sequences
                    else:
                        seqs = output
                    
                    # 获取 NFE
                    if hasattr(output, 'profile') and output.profile:
                        nfe = output.profile.num_forward_evals
                    else:
                        nfe = 0
                        
                except Exception as e:
                    print(f"\n[Error] Generation failed: {e}")
                    # fallback 空输出
                    seqs = torch.zeros((current_bs, inputs['input_ids'].shape[1] + 10), dtype=torch.long)
                    nfe = 0
                
                torch.cuda.synchronize()
                t_cost = time.perf_counter() - t_gen_start
                
                # 3. 解码与记录
                prompt_len = inputs['input_ids'].shape[1]
                preds = [
                    tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    for seq in seqs
                ]

                stats_storage[(strength, rank)]['total_time'] += t_cost
                stats_storage[(strength, rank)]['total_nfe'] += nfe
                stats_storage[(strength, rank)]['predictions'].extend(preds)
                stats_storage[(strength, rank)]['references'].extend(batch_targets_raw)
                
                if strength == 1.0:
                    baseline_results = {'time': t_cost, 'nfe': nfe, 'preds': preds}

                if args.verbose and batch_idx == 0:
                    print(f"      [Str={strength}, Rank={rank}] NFE={nfe}, Time={t_cost:.2f}s")

    # 7. 评估
    print("\n" + "=" * 90)
    print(f"{'全维度评估结果 (Strength x Rank) - APD Mode':^90}")
    print("=" * 90)

    metric_objs = load_evaluation_metrics(args.dataset)
    final_results = {}
    all_metric_keys = set()

    for key, stats in stats_storage.items():
        s, r = key
        if NUM_TEST_EXAMPLES > 0:
            metrics = compute_metrics(stats['predictions'], stats['references'], args.dataset, metric_objs)
            row = {
                'Time(s)': stats['total_time'] / NUM_TEST_EXAMPLES,
                'NFE': stats['total_nfe'] / NUM_TEST_EXAMPLES if NUM_TEST_EXAMPLES > 0 else 0
            }
            row.update(metrics)
            final_results[key] = row
            all_metric_keys.update(metrics.keys())
            
            # Save results
            if args.save_result:
                os.makedirs("results_apd", exist_ok=True)
                fname = f"results_apd/{args.dataset}_str{int(s*100)}_rank{r}.txt"
                with open(fname, 'w', encoding='utf-8') as f:
                    for p in stats['predictions']:
                        f.write(p.replace('\n', ' ') + '\n')

    # 打印表格
    sorted_keys = sorted(final_results.keys(), key=lambda x: (-x[0], x[1]))
    metric_headers = sorted(list(all_metric_keys))
    headers = ['Strength', 'Rank', 'Time(s)', 'NFE'] + metric_headers
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
        curr_str = s
        data = final_results[(s, r)]
        cells = [f"{s*100:.0f}%", f"Top-{r+1}"]
        cells.append(f"{data.get('Time(s)',0):.3f}")
        cells.append(f"{data.get('NFE',0):.1f}")

        for mh in metric_headers:
            val = data.get(mh, 0)
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        print_row(cells)

    print("└" + "┴".join(["─" * (col_w + 2) for _ in headers]) + "┘")
    print("Done.")

if __name__ == "__main__":
    main()