import re
import numpy as np
from typing import List, Dict, Callable, Any, Union
from collections import Counter
import string
from mauve import compute_mauve
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# 辅助函数
# ============================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_single(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def extract_label(text: str) -> str:
    """
    辅助函数：从文本中提取首字母选项 (A, B, C, D)。
    兼容格式："A.", "A)", "A ", "Answer: A" 等。
    """
    if not text:
        return ""
    pattern = r"(?:^|Answer:\s*)([A-D])(?:\.|:|\)|$|\s)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


# ============================================================
# 各指标的独立计算方法
# ============================================================

def calc_em(predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs) -> float:
    """计算 Exact Match (EM) 指标"""
    em_scores = []
    for pred, golds in zip(predictions, references):
        if isinstance(golds, list):
            em_scores.append(metric_max_over_ground_truths(exact_match_score, pred, golds))
        else:
            em_scores.append(1.0 if exact_match_score(pred, golds) else 0.0)
    return np.mean(em_scores)


def calc_f1(predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs) -> float:
    """计算 F1 指标"""
    f1_scores = []
    for pred, golds in zip(predictions, references):
        if isinstance(golds, list):
            f1_scores.append(metric_max_over_ground_truths(f1_score_single, pred, golds))
        else:
            f1_scores.append(f1_score_single(pred, golds))
    return np.mean(f1_scores)


def calc_rouge_l(predictions: List[str], references: Union[List[str], List[List[str]]], metric_objs: Dict = None, **kwargs) -> float:
    """计算 ROUGE-L 指标"""
    if metric_objs and 'rouge' in metric_objs:
        rouge_res = metric_objs['rouge'].compute(predictions=predictions, references=references)
        return rouge_res['rougeL']
    return 0.0


def calc_bleu(predictions: List[str], references: Union[List[str], List[List[str]]], metric_objs: Dict = None, **kwargs) -> float:
    """计算 BLEU 指标"""
    if metric_objs and 'bleu' in metric_objs:
        bleu_res = metric_objs['bleu'].compute(predictions=predictions, references=references)
        return bleu_res['bleu']
    return 0.0


def calc_meteor(predictions: List[str], references: Union[List[str], List[List[str]]], metric_objs: Dict = None, **kwargs) -> float:
    """计算 METEOR 指标"""
    if metric_objs and 'meteor' in metric_objs:
        meteor_res = metric_objs['meteor'].compute(predictions=predictions, references=references)
        return meteor_res['meteor']
    return 0.0


def calc_bertscore(predictions: List[str], references: Union[List[str], List[List[str]]], metric_objs: Dict = None, device: str = 'cuda', **kwargs) -> float:
    """计算 BERTScore 指标"""
    if metric_objs and 'bertscore' in metric_objs:
        try:
            bert_res = metric_objs['bertscore'].compute(
                predictions=predictions, references=references, 
                lang="en", device=device, batch_size=32
            )
            return np.mean(bert_res['f1'])
        except Exception as e:
            print(f"[Warning] BERTScore calculation failed: {e}")
    return 0.0


def calc_dist_1(predictions: List[str], references: List[str] = None, **kwargs) -> float:
    """计算 Distinct-1 指标"""
    if not predictions:
        return 0.0
    total_ngrams = 0
    unique_ngrams = set()
    for sentence in predictions:
        tokens = sentence.lower().strip().split()
        if len(tokens) < 1:
            continue
        total_ngrams += len(tokens)
        for token in tokens:
            unique_ngrams.add((token,))
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def calc_dist_2(predictions: List[str], references: List[str] = None, **kwargs) -> float:
    """计算 Distinct-2 指标"""
    if not predictions:
        return 0.0
    total_ngrams = 0
    unique_ngrams = set()
    for sentence in predictions:
        tokens = sentence.lower().strip().split()
        if len(tokens) < 2:
            continue
        ngrams = list(zip(tokens[:-1], tokens[1:]))
        total_ngrams += len(ngrams)
        for ngram in ngrams:
            unique_ngrams.add(ngram)
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def calc_self_bleu(predictions: List[str], references: List[str] = None, n_gram: int = 4, sample_size: int = 1000, **kwargs) -> float:
    """计算 Self-BLEU 指标"""
    if not predictions:
        return 0.0
    
    tokens_list = [text.strip().split() for text in predictions]
    
    if len(tokens_list) > sample_size:
        import random
        tokens_list = random.sample(tokens_list, sample_size)
    
    scores = []
    weight = tuple((1. / n_gram for _ in range(n_gram)))
    smoothing = SmoothingFunction().method1
    
    for i, hypothesis in enumerate(tokens_list):
        refs = tokens_list[:i] + tokens_list[i+1:]
        if not refs:
            continue
        score = sentence_bleu(refs, hypothesis, weights=weight, smoothing_function=smoothing)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def calc_gen_ppl(predictions: List[str], references: List[str] = None, 
                 model_id: str = '/root/autodl-tmp/model/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa', 
                 device: str = 'cuda', **kwargs) -> float:
    """计算生成困惑度 (Gen-PPL)"""
    if not predictions:
        return 0.0
    
    try:
        print(f"Loading Oracle Model for PPL: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()

        nlls = []
        
        for text in tqdm(predictions, desc="Calculating Gen-PPL"):
            if not text.strip():
                continue
            
            encodings = tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(device)
            
            if input_ids.size(1) < 2:
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)

        if not nlls:
            return 0.0

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    except Exception as e:
        print(f"[Warning] Gen-PPL calculation failed: {e}")
        return 0.0


def calc_mauve(predictions: List[str], references: List[str], device: str = 'cuda', **kwargs) -> float:
    """计算 MAUVE 指标"""
    try:
        if isinstance(references[0], list):
            ref_texts = [ref[0] if ref else "" for ref in references]
        else:
            ref_texts = references
        
        valid_pairs = [(p, r) for p, r in zip(predictions, ref_texts) if p.strip() and r.strip()]
        
        if len(valid_pairs) < 100:
            print(f"[Warning] MAUVE 样本数较少 ({len(valid_pairs)} 条)，结果可能不稳定")
        
        if len(valid_pairs) > 0:
            valid_preds, valid_refs = zip(*valid_pairs)
            device_id = 0 if 'cuda' in device else -1
            
            mauve_out = compute_mauve(
                p_text=list(valid_refs),
                q_text=list(valid_preds),
                featurize_model_name='/root/autodl-tmp/model/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa',
                device_id=device_id,
                max_text_length=256,
                verbose=False
            )
            return mauve_out.mauve
        else:
            print("[Warning] 无有效样本，跳过 MAUVE 计算")
            return 0.0
    except Exception as e:
        print(f"[Warning] MAUVE 计算失败: {e}")
        return 0.0


def calc_accuracy(predictions: List[str], references: List[str], dataset_name: str = '', **kwargs) -> float:
    """计算多选题准确率 (Accuracy)"""
    # GPQA 特殊处理：取最后一行
    if dataset_name == 'gpqa':
        predictions = [pred.strip().splitlines()[-1] for pred in predictions]
        references = [ref.strip().splitlines()[-1] for ref in references]
    
    correct_count = 0
    total_count = len(predictions)
    
    for pred, ref in zip(predictions, references):
        print(f"预测值: {pred}, 真实值: {ref}")
        p_label = extract_label(pred)
        r_label = extract_label(ref)
        
        if p_label and r_label:
            if p_label == r_label:
                correct_count += 1
        elif not r_label:
            clean_p = pred.strip().lower()
            clean_r = ref.strip().lower()
            if clean_r in clean_p:
                correct_count += 1
        else:
            ref_text_body = re.sub(r"^[A-D][\.\)\s]+", "", ref).strip().lower()
            pred_text_body = re.sub(r"^[A-D][\.\)\s]+", "", pred).strip().lower()
            if ref_text_body and ref_text_body in pred_text_body:
                correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0



def calc_accuracy_banking77(predictions: List[str], references: List[str], **kwargs) -> float:
    """
    计算Banking77数据集的准确率。
    允许预测有解释或附加内容，只要编号或意图英文短语（忽略大小写）匹配即可判对。
    """
    def extract_index(text: str) -> str:
        # 匹配开头的数字编号，如 "11. "，允许前面有 "Answer: " 等
        match = re.search(r"(?:^|\s)(\d{1,2})[\.\: ]", text)
        if match:
            return match.group(1).zfill(2)
        return ""

    def clean_intent(text: str) -> str:
        # 去掉编号部分
        text = re.sub(r"^\s*\d{1,2}\s*[\.\: ]\s*", "", text)
        # 替换下划线为空格，移除标点（保留空格和字母数字），转小写
        text = text.replace('_', ' ').lower()
        # 只保留字母数字和空格
        text = "".join([c for c in text if c.isalnum() or c.isspace()])
        return " ".join(text.split())

    correct_count = 0
    total_count = len(predictions)
    
    for pred, ref in zip(predictions, references):
        print(f"预测值: {pred}, 真实值: {ref}")
        pred_idx = extract_index(pred)
        ref_idx = extract_index(ref)
        
        # 1. 编号匹配
        if pred_idx and ref_idx and pred_idx == ref_idx:
            correct_count += 1
            continue
            
        # 2. 意图文本匹配
        ref_intent = clean_intent(ref)
        pred_intent = clean_intent(pred)
        
        # 如果参考意图（非空）出现在预测文本中
        if ref_intent and ref_intent in pred_intent:
            correct_count += 1
            
    return correct_count / total_count if total_count > 0 else 0.0


def calc_pass_at_1(predictions: List[str], references: List[str], **kwargs) -> float:
    """计算 Pass@1(Text) 指标（简单文本匹配）"""
    matches = []
    for pred, ref in zip(predictions, references):
        p = pred.strip()
        r = ref.strip()
        matches.append(1.0 if p == r else 0.0)
    return np.mean(matches)


# ============================================================
# 指标名称到计算方法的映射
# ============================================================

METRIC_FUNCTIONS: Dict[str, Callable] = {
    'EM': calc_em,
    'F1': calc_f1,
    'ROUGE-L': calc_rouge_l,
    'BLEU': calc_bleu,
    'METEOR': calc_meteor,
    'BERTScore': calc_bertscore,
    'Dist-1': calc_dist_1,
    'Dist-2': calc_dist_2,
    'Self-BLEU': calc_self_bleu,
    'Gen-PPL': calc_gen_ppl,
    'MAUVE': calc_mauve,
    'Accuracy': calc_accuracy,
    'Pass@1(Text)': calc_pass_at_1,
    'Accuracy_Banking77': calc_accuracy_banking77,
}


# ============================================================
# 数据集到指标列表的映射
# ============================================================

DATASET_METRICS: Dict[str, List[str]] = {
    'nq_open': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'webquestions': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'commongen': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'ms_marco': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'wmt14': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'dailymail': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'bitext_customer': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'ecommerce': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'multiwoz': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'dailydialog': ['ROUGE-L', 'BLEU', 'METEOR', 'BERTScore', 'Dist-1', 'Dist-2', 'Self-BLEU', 'Gen-PPL', 'MAUVE'],
    'e2e': ['ROUGE-L', 'BLEU', 'METEOR', 'BERTScore'],
    'arc': ['Accuracy'],
    'hellaswag': ['Accuracy'],
    'mmlu': ['Accuracy'],
    'gpqa': ['Accuracy'],
    'commonsenseqa': ['Accuracy'],
    'piqa': ['Accuracy'],
    'medqa': ['Accuracy'],
    'banking77':['Accuracy_Banking77'],
    'mbpp': ['Pass@1(Text)'],
    'humaneval': ['Pass@1(Text)'],
    'spider': ['Pass@1(Text)','EM'],
    'math': ['Pass@1(Text)'],
    'gsm8k': ['Pass@1(Text)'],
    'wikitext': ['Gen-PPL', 'MAUVE', 'Dist-1', 'Dist-2'],
    'lm1b': ['Gen-PPL', 'MAUVE', 'Dist-1', 'Dist-2'],
    'rocstories': ['ROUGE-L', 'BERTScore', 'Gen-PPL', 'MAUVE', 'Dist-1', 'Dist-2'],
    'qqp': ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore', 'Gen-PPL','MAUVE'],
    'alpaca': ['ROUGE-L', 'BLEU', 'BERTScore', 'Dist-1', 'Dist-2', 'Gen-PPL', 'MAUVE'],
    'openorca': ['ROUGE-L', 'BLEU', 'BERTScore', 'Dist-1', 'Dist-2', 'Gen-PPL', 'MAUVE'],
    'dolly': ['ROUGE-L', 'BLEU', 'BERTScore', 'Dist-1', 'Dist-2', 'Gen-PPL', 'MAUVE'],
    'sharegpt': ['ROUGE-L', 'BLEU', 'BERTScore', 'Dist-1', 'Dist-2', 'Gen-PPL', 'MAUVE'],
    # 测试用：包含所有指标
    'test': ['EM', 'F1', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore', 'Dist-1', 'Dist-2', 'Self-BLEU', 'Gen-PPL', 'MAUVE', 'Accuracy', 'Pass@1(Text)'],
}


# ============================================================
# 单一指标计算入口
# ============================================================

def compute_single_metric(
    metric_name: str,
    predictions: List[str],
    references: List[str],
    metric_objs: Dict = None,
    device: str = 'cuda',
    dataset_name: str = '',
    **kwargs
) -> float:
    """
    通用的单一指标计算方法
    
    Args:
        metric_name: 指标名称
        predictions: 预测结果列表
        references: 参考答案列表
        metric_objs: evaluate 库的指标对象字典
        device: 设备
        dataset_name: 数据集名称（某些指标需要）
        **kwargs: 其他参数
    
    Returns:
        指标分数
    """
    if metric_name not in METRIC_FUNCTIONS:
        print(f"[Warning] 未知指标: {metric_name}")
        return 0.0
    
    metric_fn = METRIC_FUNCTIONS[metric_name]
    
    return metric_fn(
        predictions=predictions,
        references=references,
        metric_objs=metric_objs,
        device=device,
        dataset_name=dataset_name,
        **kwargs
    )


# ============================================================
# 主计算函数
# ============================================================

def compute_metrics(
    predictions: List[str], 
    references: List[str], 
    dataset_name: str, 
    metric_objs: Dict, 
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    根据数据集类型自动选择并计算评估指标。
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
        dataset_name: 数据集名称
        metric_objs: evaluate 库的指标对象字典
        device: 设备
    
    Returns:
        指标结果字典
    """
    results = {}
    dataset_name_lower = dataset_name.lower()
    
    # 针对 nq_open 和 webquestions，把参考答案从字符串拆分为 list
    if dataset_name_lower in ['nq_open', 'webquestions']:
        references = [ref.split("||") if isinstance(ref, str) else ref for ref in references]
        references = [[ans.strip() for ans in ref] for ref in references]
        print(f"预测值:{predictions}\n真实值:{references}")
    
    # 获取该数据集对应的指标列表
    metric_list = DATASET_METRICS.get(dataset_name_lower, [])
    
    if not metric_list:
        print(f"[Warning] 未找到数据集 '{dataset_name}' 的指标配置")
        return results
    
    # 遍历指标列表，依次计算每个指标
    for metric_name in metric_list:
        try:
            score = compute_single_metric(
                metric_name=metric_name,
                predictions=predictions,
                references=references,
                metric_objs=metric_objs,
                device=device,
                dataset_name=dataset_name_lower
            )
            results[metric_name] = score
        except Exception as e:
            print(f"[Warning] 指标 '{metric_name}' 计算失败: {e}")
            results[metric_name] = 0.0
    
    return results









if __name__ == "__main__":
    # import evaluate
    
    # print("\n" + "=" * 80)
    # print("          评估指标模块测试 (eval2.py)")
    # print("=" * 80)
    
    # # ============================================================
    # # 测试数据准备
    # # ============================================================
    # test_predictions = [
    #     "The capital of France is Paris.",
    #     "Machine learning is a subset of artificial intelligence.",
    #     "Python is a programming language.",
    #     "A. The answer is correct.",
    #     "B. This is option B.",
    # ]
    # test_references = [
    #     "The capital of France is Paris.",
    #     "Machine learning is part of AI.",
    #     "Python is a popular programming language.",
    #     "A. The answer is correct.",
    #     "C. This is option C.",
    # ]
    
    # print(f"\n[测试数据] 预测样本数: {len(test_predictions)}, 参考样本数: {len(test_references)}")
    
    # # ============================================================
    # # 加载 evaluate 库的指标对象
    # # ============================================================
    # print("\n[加载评估指标对象]")
    # metric_objs = {}
    # for name in ['rouge', 'bleu', 'meteor', 'bertscore']:
    #     try:
    #         metric_objs[name] = evaluate.load(name)
    #         print(f"  ✓ {name.upper()} 加载成功")
    #     except Exception as e:
    #         print(f"  ✗ {name.upper()} 加载失败: {e}")
    
    # # ============================================================
    # # 使用 'test' 数据集测试所有指标
    # # ============================================================
    # print("\n" + "=" * 80)
    # print(f"  使用 'test' 数据集测试 (包含所有 {len(DATASET_METRICS['test'])} 个指标)")
    # print("=" * 80)
    
    # results = compute_metrics(
    #     predictions=test_predictions,
    #     references=test_references,
    #     dataset_name='test',
    #     metric_objs=metric_objs,
    #     device='cpu'
    # )
    
    # # 美化打印结果表格
    # headers = ['Metric', 'Value']
    # header_row = "| " + " | ".join([f"{h:<20}" for h in headers]) + " |"
    # separator = "|" + "|".join(["-" * 22 for _ in headers]) + "|"
    
    # print(separator)
    # print(header_row)
    # print(separator)
    
    # for metric_name, value in results.items():
    #     if isinstance(value, float):
    #         print(f"| {metric_name:<20} | {value:<20.4f} |")
    #     else:
    #         print(f"| {metric_name:<20} | {str(value):<20} |")
    
    # print(separator)
    # print("\n" + "=" * 80)
    # print("          测试完成")
    # print("=" * 80)
    
    
    # 测试 calc_accuracy_banking77
    test_predictions = [
        "11. card_arrival",
        "23. contactless_not_working",
        "05. automatic_top_up",
        "42. pending card pay11ment",
        "11. card_arrival"
    ]
    test_references = [
        "11. card_arrival",
        "23. contactless_not_working",
        "05. automatic_top_up",
        "41. pending card payment",
        "12. card_delivery_estimate"
    ]
    acc = calc_accuracy_banking77(test_predictions, test_references)
    print(f"[Banking77 Accuracy] {acc:.4f}")