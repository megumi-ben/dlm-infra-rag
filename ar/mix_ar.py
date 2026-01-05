import time
import argparse
import os
import torch
import gc
from typing import List, Dict

# 引入项目模块 (保持原有引用)
from rag_engine import AdvancedRAGEngine
from dataset_processor import (
    process_prompt, 
    split_tarin_test, 
    load_evaluation_metrics
)
from evaluation import compute_metrics

def construct_rag_prompt(original_prompt: str, retrieved_context: str) -> str:
    """
    构建 RAG Prompt。
    将检索到的 Context (Prompt + Target) 作为参考信息放在前面。
    """
    if not retrieved_context:
        return original_prompt
    
    # 构建模板：
    # References:
    # <Retrieved Q&A Pair 1>
    # <Retrieved Q&A Pair 2>
    # ...
    # 
    # <Current Processed Prompt>
    
    return f"Examaples:\n{retrieved_context}\nReal Questions:\n{original_prompt}"

def main():
    parser = argparse.ArgumentParser(description="运行 AR Baseline (vLLM + RAG) 对比实验 - Enhanced Version")
    
    # --- 基础配置 ---
    parser.add_argument('--model_path', type=str, default='/backup01/DLM/model/Qwen2.5-7B-Instruct', help='AR模型路径')
    parser.add_argument('--dataset', type=str, default='mmlu', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--num_test_examples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--save_result', action='store_true', help='是否保存结果文件')
    
    # --- RAG 配置 ---
    parser.add_argument('--embed_model', type=str, default='BAAI/bge-small-zh-v1.5', help='Embedding 模型路径')
    parser.add_argument('--rerank_model', type=str, default='BAAI/bge-reranker-v2-m3', help='Reranker 模型路径')
    parser.add_argument('--test_top_k', type=int, default=1, help='检索 Top-K 深度')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='数据使用比例')
    parser.add_argument('--max_size', type=int, default=None, help='数据最大条数')
    parser.add_argument('--batch_size', type=int, default=-1, help='批量大小')
    parser.add_argument('--rag_engine', type=str, default='ours', help='RAG引擎类型 (ours/flashrag)')

    # --- 生成配置 (vLLM) ---
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-P')
    parser.add_argument('--max_tokens', type=int, default=512, help='最大生成长度')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='vLLM 显存占用比例')
    
    # --- 高级特性配置 (新增) ---
    parser.add_argument('--use_flashinfer', action='store_true', help='是否启用 FlashInfer 后端')
    parser.add_argument('--quantization', type=str, default=None, help='量化方法, e.g., "awq"')
    parser.add_argument('--draft_model_path', type=str, default=None, help='投机解码(EAGLE) Draft Model 路径')
    parser.add_argument('--num_speculative_tokens', type=int, default=5, help='投机解码步数')

    # --- 实验模式 ---
    parser.add_argument('--compare_mode', type=str, default='rank_specific', 
                        choices=['rank_specific', 'concat'],
                        help='RAG模式: rank_specific(对比mix.py, 分别测试Rank0, Rank1...) 或 concat(拼接TopK所有内容)')

    args = parser.parse_args()

    # ========================================================================
    # 0. 环境配置 (必须在 import vllm 之前设置)
    # ========================================================================
    if args.use_flashinfer:
        print(">>> 启用 FlashInfer 后端...")
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    
    # 延迟引入 vLLM，确保环境变量生效
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vLLM library is not installed. Please install it to run this AR baseline.")
        exit(1)

    print(f"\n" + "=" * 80)
    print(f"{'MAIN_AR.PY - AR Baseline (vLLM + RAG + Features)':^80}")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    if args.draft_model_path:
        print(f"投机解码 (EAGLE): {args.draft_model_path}")
    if args.quantization:
        print(f"量化模式: {args.quantization}")
    if args.use_flashinfer:
        print(f"加速后端: FlashInfer")
    print(f"数据集: {args.dataset} | 样本数: {args.num_test_examples}")
    print("=" * 80 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========================================================================
    # 1. 准备数据
    # ========================================================================
    print(">>> 正在加载数据...")
    train_prompts, train_targets, test_prompts, test_targets = split_tarin_test(
        args.dataset, args.data_path, args.data_ratio, args.max_size
    )
    
    # 截断测试集
    if args.num_test_examples > len(test_prompts):
        args.num_test_examples = len(test_prompts)
    
    eval_prompts = test_prompts[:args.num_test_examples]
    eval_targets = test_targets[:args.num_test_examples]
    print(f"数据准备完成: Train(库)={len(train_prompts)}, Test(评估)={len(eval_prompts)}")

    # ========================================================================
    # 2. 执行 RAG 检索 (Search First Strategy)
    # ========================================================================
    print("\n>>> [Phase 1] 启动 RAG 引擎进行检索...")
    retrieval_cache = [] # 存储处理后的检索文本
    
    try:
        if args.rag_engine == 'flashrag':
            from rag2 import FlashRAGEngine
            rag = FlashRAGEngine()
        else:
            rag = AdvancedRAGEngine(
                embed_model=args.embed_model, 
                rerank_model=args.rerank_model, 
                device=device
            )
        
        # 构建知识库
        rag.build_knowledge_base(train_prompts, train_targets, batch_size=256)
        
        # 批量检索
        print(f"正在为 {len(eval_prompts)} 条测试数据检索 Top-{args.test_top_k}...")
        start_search = time.perf_counter()
        
        # retrieval_results 是一个 List[List[Dict]]
        retrieval_results, _ = rag.batch_search(
            eval_prompts,
            top_k_retrieve=args.test_top_k * 2,
            top_n_rerank=args.test_top_k,
            batch_size=32
        )
        elapsed_search = time.perf_counter() - start_search
        print(f"检索完成，耗时: {elapsed_search:.2f}s")
        
        # 处理检索结果
        for res_list in retrieval_results:
            processed_list = []
            for res in res_list:
                r_prompt = res['prompt'].strip()
                r_target = res['target'].strip()
                combined_text = f"{r_prompt}\nThe correct answer: {r_target}"
                processed_list.append(combined_text)
            retrieval_cache.append(processed_list)
            
        # 释放显存
        del rag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print(">>> [Phase 1] RAG 阶段结束，显存已清理。")
        
    except Exception as e:
        print(f"RAG 阶段发生错误: {e}")
        exit(1)

    # ========================================================================
    # 3. 加载 vLLM 模型 (含 EAGLE, AWQ, FlashInfer配置)
    # ========================================================================
    print("\n>>> [Phase 2] 加载 AR 模型 (vLLM)...")
    try:
        # 配置投机解码 (EAGLE)
        speculative_config = None
        if args.draft_model_path:
            speculative_config = {
                "method": "eagle",
                "model": args.draft_model_path,
                "num_speculative_tokens": args.num_speculative_tokens,
            }

        # 确定 dtype
        # 如果使用 AWQ 量化，通常建议 dtype="auto"
        # 否则默认使用 bfloat16
        dtype_setting = "bfloat16"
        if args.quantization and "awq" in args.quantization.lower():
            dtype_setting = "auto"
        
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=4096,
            tensor_parallel_size=1,
            dtype=dtype_setting,
            quantization=args.quantization,     # 支持 AWQ
            speculative_config=speculative_config # 支持 EAGLE
        )
        tokenizer = llm.get_tokenizer()
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop_token_ids=[tokenizer.eos_token_id]
        )
    except Exception as e:
        print(f"vLLM 加载失败: {e}")
        exit(1)

    # ========================================================================
    # 4. 实验循环
    # ========================================================================
    # -1: Baseline (No RAG)
    # 0..K: RAG (Rank 0..K)
    test_configs = [-1] + list(range(args.test_top_k))
    
    stats_storage = {} 
    print("正在加载评估指标...")
    metric_objs = load_evaluation_metrics(args.dataset)

    for config_idx in test_configs:
        current_prompts = []
        
        # --- 构造 Prompt ---
        if config_idx == -1:
            mode_name = "Baseline (Zero-shot)"
            for raw_p in eval_prompts:
                final_p = process_prompt(raw_p, args.dataset)
                current_prompts.append(final_p)
        else:
            if args.compare_mode == 'rank_specific':
                mode_name = f"RAG (Rank {config_idx})"
                for i, raw_p in enumerate(eval_prompts):
                    retrieved_texts = retrieval_cache[i]
                    context = retrieved_texts[config_idx] if len(retrieved_texts) > config_idx else ""
                    base_p = process_prompt(raw_p, args.dataset)
                    final_p = construct_rag_prompt(base_p, context)
                    current_prompts.append(final_p)
            else:
                mode_name = f"RAG (Concat Top-{config_idx+1})"
                for i, raw_p in enumerate(eval_prompts):
                    retrieved_texts = retrieval_cache[i]
                    concat_context = "\n\n".join(retrieved_texts[:config_idx+1])
                    base_p = process_prompt(raw_p, args.dataset)
                    final_p = construct_rag_prompt(base_p, concat_context)
                    current_prompts.append(final_p)

        print(f"\n正在进行推理: [{mode_name}] ...")
        
        # --- Chat Template 格式化 ---
        chat_prompts = []
        for p in current_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_prompts.append(text)
        for i, p in enumerate(chat_prompts):
            print(f"Prompt {i}:\n{p}\n{'-'*30}")
            
        # --- vLLM 批量推理 ---
        batch_size=args.batch_size if args.batch_size else -1
        if batch_size<0:
            # vLLM 自动处理 Continuous Batching，无需手动并行循环
            start_gen = time.perf_counter()
            outputs = llm.generate(chat_prompts, sampling_params, use_tqdm=True)
            end_gen = time.perf_counter()
            total_time = end_gen - start_gen
            # --- 提取结果 ---
            generated_texts = [output.outputs[0].text.strip() for output in outputs]
        else:
            generated_texts = []
            total_time = 0
            for batch_start in range(0, len(chat_prompts), batch_size):
                batch_prompts = chat_prompts[batch_start:batch_start + batch_size]
                start_gen = time.perf_counter()
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                end_gen = time.perf_counter()
                total_time += (end_gen - start_gen)
                generated_texts.extend([output.outputs[0].text.strip() for output in outputs])


        avg_time = total_time / len(chat_prompts)
        print(f"  -> 样例输入: {chat_prompts[0][-200:]}...") 
        print(f"  -> 样例输出: {generated_texts[0][:100]}...")
        print(f"  -> 平均耗时: {avg_time:.4f}s (Total: {total_time:.2f}s)")
        
        # --- 存储 ---
        stats_storage[config_idx] = {
            'mode_name': mode_name,
            'predictions': generated_texts,
            'references': eval_targets,
            'avg_time': avg_time
        }

    # ========================================================================
    # 5. 评估与输出
    # ========================================================================
    print("\n" + "=" * 90)
    print(f"{'AR Baseline 评估结果':^90}")
    print("=" * 90)
    
    final_results = {}
    all_metric_keys = set()
    
    for idx, stats in stats_storage.items():
        print(f"正在评估: {stats['mode_name']} ...")
        
        metrics = compute_metrics(
            stats['predictions'], 
            stats['references'], 
            args.dataset, 
            metric_objs, 
            device=device
        )
        
        row = {'Time(s)': stats['avg_time']}
        row.update(metrics)
        final_results[idx] = row
        all_metric_keys.update(metrics.keys())
        
        if args.save_result:
            os.makedirs("./results_ar", exist_ok=True)
            fname = f"./results_ar/{args.dataset}_ar_{idx}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                for line in stats['predictions']:
                    f.write(line.replace('\n', ' ') + '\n')

    # --- 打印表格 ---
    metric_headers = sorted(list(all_metric_keys))
    headers = ['Config', 'Time(s)'] + metric_headers
    col_w = 15
    
    def print_row(vals, is_header=False):
        row_s = "│ " + " │ ".join([f"{str(v):^{col_w}}" for v in vals]) + " │"
        print(row_s)
        if is_header:
            print("├" + "┼".join(["─" * (col_w + 2) for _ in vals]) + "┤")
            
    print("\n┌" + "┬".join(["─" * (col_w + 2) for _ in headers]) + "┐")
    print_row(headers, is_header=True)
    
    for idx in test_configs:
        data = final_results[idx]
        mode_str = stats_storage[idx]['mode_name']
        if len(mode_str) > col_w: 
            mode_str = mode_str[:col_w-3] + "..."
            
        cells = [mode_str]
        cells.append(f"{data.get('Time(s)', 0):.4f}")
        
        for mh in metric_headers:
            val = data.get(mh, 0)
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            
        print_row(cells)
        
    print("└" + "┴".join(["─" * (col_w + 2) for _ in headers]) + "┘")

if __name__ == "__main__":
    main()