import os
import sys
import time
import gc
import argparse
import random
import torch
import types
import numpy as np

from transformers import AutoTokenizer
from model.modeling_dream import DreamModel

# # 添加 temp 目录到路径，以便导入评估相关模块
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../temp'))

from dataset_processor import (
	process_prompt,
	split_tarin_test,
	load_evaluation_metrics
)
from evaluation import compute_metrics

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
    parser = argparse.ArgumentParser(description="Fast-dLLM Dream 评估脚本")
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/model/Dream-v0-Instruct-7B', help='模型路径')
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--steps', type=int, default=128, help='扩散步数')
    parser.add_argument('--block_length', type=int, default=32, help='Block 长度')
    parser.add_argument('--temperature', type=float, default=0., help='采样温度')
    parser.add_argument('--remasking', type=str, default='low_confidence', choices=['low_confidence', 'random'], help='重掩码策略')
    parser.add_argument('--use_cache', action='store_true', help='启用 Prefix KV Cache')
    parser.add_argument('--dual_cache', action='store_true', help='启用 Dual Cache (更高效)')
    parser.add_argument('--threshold', type=float, default=None, help='置信度阈值 (并行生成)')
    parser.add_argument('--dataset', type=str, default='arc', help='数据集名称: mmlu, arc, gpqa, math, gsm8k, hellaswag, piqa, nq_open, etc.')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/datasets', help='本地数据集根目录')
    parser.add_argument('--num_test_examples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--batch_size', type=int, default=4, help='批量推理大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--save_result', action='store_true', help='保存结果到文件')
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 80)
    print(f"{'Fast-dLLM Dream 评估实验':^80}")
    print("=" * 80)
    print(f"数据集: {args.dataset.upper()}")
    print(f"模型: {args.model_path}")
    print(f"生成参数: steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length}")
    print(f"加速策略: use_cache={args.use_cache}, dual_cache={args.dual_cache}, threshold={args.threshold}")
    print(f"温度: {args.temperature}, 重掩码: {args.remasking}")
    print("=" * 80 + "\n")

    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    print("模型和分词器加载完毕。")
    
    # cache相关
    if args.use_cache:
        from model.generation_utils_block import DreamGenerationMixin
        model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
        model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    else:
        from model.generation_utils import DreamGenerationMixin
        model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
        model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    
    # dual_cache 模式下强制 batch_size=1
    if args.dual_cache:
        if args.batch_size != 1:
            print(f"[警告] dual_cache 模式不支持批量推理，batch_size 已从 {args.batch_size} 强制设为 1")
            args.batch_size = 1

    print(f"\n正在加载数据集: {args.dataset}...")
    train_prompts, train_targets, test_prompts, test_targets = split_tarin_test(
        args.dataset, args.data_path
    )
    print(f"划分完成: Train(RAG库)={len(train_prompts)}, Test(评估)={len(test_prompts)}")

    NUM_TEST_EXAMPLES = min(args.num_test_examples, len(test_prompts))
    batch_size = args.batch_size

    stats = {
        'total_time': 0,
        'total_nfe': 0,
        'predictions': [],
        'references': []
    }

    print("\n[GPU 预热] 正在执行...")
    warmup_text = test_prompts[0] if test_prompts else "Hello, world!"
    warmup_msg = [{"role": "user", "content": process_prompt(warmup_text, args.dataset)}]
    warmup_prompt = tokenizer.apply_chat_template(warmup_msg, add_generation_prompt=True, tokenize=False)
    warmup_ids = tokenizer(warmup_prompt, return_tensors="pt")['input_ids'].to(device)
    warmup_attention_mask = torch.ones_like(warmup_ids)

    # 预热
    _ = model.diffusion_generate(
        warmup_ids,
        attention_mask=warmup_attention_mask,
        max_new_tokens=32,
        steps=32,
        block_length=32,
        top_p=args.top_p,
        alg=args.alg,
        alg_temp=args.alg_temp,
        top_k=args.top_k,
        temperature=args.temperature,
        threshold=args.threshold,
        dual_cache=args.dual_cache
    )
    print("[GPU 预热] ✓ 完成\n")

    num_batches = (NUM_TEST_EXAMPLES + batch_size - 1) // batch_size

    print("=" * 70)
    print(f"{'批量推理开始':^70}")
    print(f"总样本: {NUM_TEST_EXAMPLES} | 批次大小: {batch_size} | 总批次: {num_batches}")
    print("=" * 70)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, NUM_TEST_EXAMPLES)
        current_batch_size = batch_end - batch_start

        batch_prompts = test_prompts[batch_start:batch_end]
        batch_targets = test_targets[batch_start:batch_end]

        print(f"\n┌{'─'*68}┐")
        print(f"│ 批次 {batch_idx+1}/{num_batches} │ 样本 {batch_start+1}-{batch_end} / {NUM_TEST_EXAMPLES} │")
        print(f"└{'─'*68}┘")

        batch_messages = [
            {"role": "user", "content": process_prompt(prompt_text, args.dataset)}
            for prompt_text in batch_prompts
        ]
        batch_formatted_prompts = [
            tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
            for msg in batch_messages
        ]
        encoded_outputs = tokenizer(
            batch_formatted_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        attention_mask = encoded_outputs['attention_mask'].to(device)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.gen_length,
            return_dict_in_generate=True,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            threshold=args.threshold,
            dual_cache=args.dual_cache,
            use_cache=args.use_cache
        )

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        # 解码输出
        generations = [
            tokenizer.decode(seq[len(input_ids[i]):].tolist(), skip_special_tokens=True)
            for i, seq in enumerate(output.sequences)
        ]

        stats['total_time'] += elapsed
        stats['total_nfe'] += getattr(output, 'nfe', 0)
        stats['predictions'].extend(generations)
        stats['references'].extend(batch_targets)

        print(f"  Time: {elapsed:.2f}s | NFE: {getattr(output, 'nfe', 0)}")
        if args.verbose:
            for i, (pred, ref) in enumerate(zip(generations, batch_targets)):
                print(f"  [{batch_start+i+1}] Pred: {pred[:100]}...")
                print(f"       Ref:  {ref[:100]}...")

        del input_ids, attention_mask, output
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'评估阶段':^70}")
    print("=" * 70)

    metric_objs = load_evaluation_metrics(args.dataset)
    avg_time_batch = stats['total_time'] / num_batches
    avg_time_single = stats['total_time'] / NUM_TEST_EXAMPLES
    avg_nfe = stats['total_nfe'] / num_batches
    metrics = compute_metrics(
        stats['predictions'],
        stats['references'],
        args.dataset,
        metric_objs
    )

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print(f"║{'实 验 结 果 汇 总':^76}║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  数据集: {args.dataset.upper():<15} 样本数: {NUM_TEST_EXAMPLES:<8} 批大小: {batch_size:<8}║")
    print(f"║  加速: cache={args.use_cache} dual={args.dual_cache} threshold={args.threshold}              ║")
    print("╚" + "═" * 78 + "╝")

    print("\n┌────────────────────────────────────────────────────────┐")
    print(f"│  总耗时: {stats['total_time']:.2f}s")
    print(f"│  平均时间 (每批次): {avg_time_batch:.2f}s")
    print(f"│  平均时间 (每样本): {avg_time_single:.2f}s")
    print(f"│  总 NFE: {stats['total_nfe']}")
    print(f"│  平均 NFE: {avg_nfe:.0f}")
    print(f"│  Steps 预算: {args.steps}")
    print(f"│  NFE 节省率: {(1 - avg_nfe/args.steps)*100:.1f}%")
    print("├────────────────────────────────────────────────────────┤")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"│  {k}: {v:.4f}")
        else:
            print(f"│  {k}: {v}")
    print("└────────────────────────────────────────────────────────┘")

    print("\n" + "─" * 70)
    print("参数说明:")
    print("  • --use_cache: 启用 Prefix KV Cache")
    print("  • --dual_cache: 启用 Dual Cache (块级替换，更高效)")
    print("  • --threshold: 置信度阈值 (动态并行生成，如 0.9)")
    print("─" * 70)
    print(f"\n实验完成! 数据集: {args.dataset.upper()} | 总样本: {NUM_TEST_EXAMPLES}")

if __name__ == '__main__':
	main()
