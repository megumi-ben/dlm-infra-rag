"""
dataset_processor.py - 数据集处理模块
支持18种数据集的问答对提取
"""

import os
from typing import List, Tuple
from datasets import load_from_disk, concatenate_datasets
import numpy as np
from typing import List, Tuple, Dict
import random
import evaluate
import time


banking77_label_names = [
                "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
                "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire",
                "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
                "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
                "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working", "country_support",
                "declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised", "disposable_card_limits",
                "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
                "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
                "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
                "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
                "receiving_money", "Refund_not_showing_up", "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
                "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
                "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
                "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
                "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
                "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
            ]

def format_choices(choices: List[str]) -> str:
    """辅助函数：将选项列表格式化为 A. xxx B. xxx ..."""
    options = ["A", "B", "C", "D", "E", "F"]
    return "\n".join([f"{opt}. {c}" for opt, c in zip(options, choices)])


def load_and_process_dataset(dataset_name: str, base_path: str) -> Tuple[List[str], List[str]]:
    """
    加载本地数据集，合并指定切分，并转换为 (Prompt, Target) 列表。
    
    Args:
        dataset_name: 数据集名称
        base_path: 数据集根目录
        
    Returns:
        (prompts, targets) 元组
        
    支持的数据集：
        mmlu, arc, gpqa, math, gsm8k, hellaswag, mbpp, humaneval, piqa,
        nq_open, commonsenseqa, webquestions, dailydialog, e2e, alpaca, dolly, spider, sharegpt
    """
    dataset_name = dataset_name.lower()
    
    # 1. 映射文件夹名称
    name_map = {
        'mmlu': 'MMLU',
        'arc': 'ARC-Challenge',
        'gpqa': 'GPQA-Diamond',
        'math': 'Math',
        'gsm8k': 'GSM8K',
        'hellaswag': 'Hellaswag',
        'mbpp': 'MBPP-full',
        'humaneval': 'Humaneval',
        'piqa': 'PIQA',
        "nq_open": "National-Questions-Open",
        "commonsenseqa": "CommonsenseQA",
        "webquestions": "WebQuestions",
        "dailydialog": "DailyDialog",
        "e2e": "E2E-NLG",
        "alpaca": "Alpaca-Cleaned",
        "dolly": "Databricks-Dolly-15k",
        "spider": "Spider",
        "sharegpt": "ShareGPT-NoSorry",
        "wikitext": "WikiText-103",
        "qqp": "QQP",
        "rocstories": "ROCStories",
        "lm1b": "LM1B",
        "banking77": "Banking77",
        "commongen": "CommonGen",
        "wmt14": "WMT14-DE-EN",
        "dailymail": "CNN-DailyMail",
        "bitext_customer": "Bitext-Customer-Support",
        "ecommerce": "E-Commerce",
        "multiwoz": "MultiWoz-2.2",
        "ms_marco": "MS-Marco",
        "medqa": "MedQA-USMLE",
        "openorca": "OpenOrca",
    }
    
    folder_name = name_map.get(dataset_name)
    if not folder_name:
        raise ValueError(f"未知的数据集名: {dataset_name}，支持的数据集: {list(name_map.keys())}")
        
    full_path = os.path.join(base_path, folder_name)
    print(f"\033[1;32m正在从本地加载: {full_path} ...\033[0m")
    
    try:
        dataset_dict = load_from_disk(full_path)
        
        # 2. 确定要合并的切分
        splits_to_merge = []
        available_splits = list(dataset_dict.keys())
        
        if dataset_name == 'piqa':
            # PIQA 特殊处理：只合并 train 和 validation，忽略 test
            for s in ['train', 'validation']:
                if s in available_splits:
                    splits_to_merge.append(s)
        elif dataset_name == 'e2e':
            # e2e 特殊处理：只合并 train 和 dev,暂时忽略 test
            for s in ['train', 'dev']:
                if s in available_splits:
                    splits_to_merge.append(s)
        else:
            splits_to_merge = available_splits
            
        print(f"正在合并切分: {splits_to_merge}")
        datasets_list = [dataset_dict[s] for s in splits_to_merge]
        if not datasets_list:
            raise ValueError("没有找到可用的切分！")
            
        merged_dataset = concatenate_datasets(datasets_list)
        print(f"合并后总数据量: {len(merged_dataset)}")

        # 3. 格式化为 (Prompt, Target)
        prompts = []
        targets = []
        
        
        if dataset_name == 'e2e':
            # E2E 需要按 mr 聚合所有 ref
            # 先收集所有数据
            # mr_to_refs = {}
            # for item in merged_dataset:
            #     mr = item.get('mr', '').strip()
            #     ref = item.get('ref', '').strip()
            #     if mr and ref:
            #         if mr not in mr_to_refs:
            #             mr_to_refs[mr] = []
            #         mr_to_refs[mr].append(ref)
            
            # # 再构造 prompts 和 targets
            # for mr, refs in mr_to_refs.items():
            #     p = f"{mr}"
            #     t = " || ".join(refs)  # 多参考答案拼接
            #     prompts.append(p)
            #     targets.append(t)
            for item in merged_dataset:
                mr = item.get('mr', '').strip()
                ref = item.get('ref', '').strip()
                p = f"{mr}"
                t = f"{ref}"
                if p and t:
                    prompts.append(p)
                    targets.append(t)
                    
                    
            test_prompts = []
            test_targets = []
            # mr_to_refs = {}
            # for item in dataset_dict['test_w_refs']:
            #     mr = item.get('mr', '').strip()
            #     ref = item.get('ref', '').strip()
            #     if mr and ref:
            #         if mr not in mr_to_refs:
            #             mr_to_refs[mr] = []
            #         mr_to_refs[mr].append(ref)
            
            # # 再构造 prompts 和 targets
            # for mr, refs in mr_to_refs.items():
            #     p = f"{mr}"
            #     t = " || ".join(refs)  # 多参考答案拼接
            #     test_prompts.append(p)
            #     test_targets.append(t)
            for item in dataset_dict['test_w_refs']:
                mr = item.get('mr', '').strip()
                ref = item.get('ref', '').strip()
                p = f"{mr}"
                t = f"{ref}"
                if p and t:
                    test_prompts.append(p)
                    test_targets.append(t)
            return prompts, targets, test_prompts, test_targets
        
        if dataset_name == "banking77":
            for item in merged_dataset:
                text = item.get("text", "").strip()
                label_idx = item.get("label", None)
                if text and label_idx is not None and 0 <= label_idx < len(banking77_label_names):
                    prompts.append(text)
                    targets.append(f"{label_idx}. {banking77_label_names[label_idx]}")
        
        elif dataset_name == "commongen":
            # 1. 按 concept_set_idx 聚合
            idx_to_targets = {}
            idx_to_concepts = {}
            for item in merged_dataset:
                idx = item.get("concept_set_idx")
                target = item.get("target", "").strip()
                concepts = item.get("concepts", [])
                if idx is not None and target:
                    if idx not in idx_to_targets:
                        idx_to_targets[idx] = []
                        idx_to_concepts[idx] = concepts
                    idx_to_targets[idx].append(target)
            prompts = []
            targets = []
            for idx in idx_to_targets:
                concept_str = " ".join(idx_to_concepts[idx])
                target_str = " || ".join(idx_to_targets[idx])
                prompts.append(concept_str)
                targets.append(target_str)
            return prompts, targets
        
        
        elif dataset_name == "bitext_customer":
            # Bitext-Customer-Support: 智能客服单轮问答
            for item in merged_dataset:
                instruction = item.get("instruction", "").strip()
                response = item.get("response", "").strip()
                if instruction and response:
                    prompts.append(instruction)
                    targets.append(response)
        
        
        elif dataset_name == "ecommerce":
            # E-Commerce: 智能客服单轮问答，utterances为多轮历史，response为客服回复
            for item in merged_dataset:
                utterances = item.get("utterances", [])
                response = item.get("response", "").strip()
                # 用角色拼接历史，假设用户A、客服B轮流
                dialog_lines = []
                for idx, utt in enumerate(utterances):
                    role = "A" if idx % 2 == 0 else "B"
                    dialog_lines.append(f"{role}: {utt.strip()}")
                prompt = "\n".join(dialog_lines)
                if prompt and response:
                    prompts.append(prompt)
                    targets.append(response)
                    
        elif dataset_name == "multiwoz":
            # MultiWoz-2.2: 单轮对话抽取，USER最后一句为prompt，SYSTEM最后一句为target
            for item in merged_dataset:
                turns = item.get("turns", [])
                # 找到最后一组 USER-SYSTEM 对
                user_utt, system_utt = None, None
                for i in range(len(turns) - 1):
                    if turns[i]["speaker"].upper() == "USER" and turns[i+1]["speaker"].upper() == "SYSTEM":
                        user_utt = turns[i]["utterance"].strip()
                        system_utt = turns[i+1]["utterance"].strip()
                if user_utt and system_utt:
                    prompts.append(user_utt)
                    targets.append(system_utt)
        
        elif dataset_name == "ms_marco":
            # 只用 query 和 answers 字段，答案用 || 拼接，若为空则用 "Answers to generate"
            for item in merged_dataset:
                query = item.get("query", "").strip()
                answers = item.get("answers", [])
                # 答案拼接
                if answers and any(a.strip() for a in answers):
                    answer = " || ".join([a.strip() for a in answers if a.strip()])
                else:
                    answer = "Answers to generate"
                if query and answer:
                    prompts.append(query)
                    targets.append(answer)
            return prompts, targets
        
        elif dataset_name == "medqa":
            # MedQA-USMLE: 单轮医学选择题，prompt为question+options，target为标准答案
            for item in merged_dataset:
                question = item.get("question", "").strip()
                options_dict = item.get("options", {})
                # 格式化选项
                options_str = "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
                prompt = f"{question}\n{options_str}"
                answer_idx = item.get("answer_idx", "").strip()
                answer = options_dict.get(answer_idx, "").strip()
                # target格式为 "C. Scorpion sting"
                if answer_idx and answer:
                    target = f"{answer_idx}. {answer}"
                else:
                    target = ""
                if prompt and target:
                    prompts.append(prompt)
                    targets.append(target)
            return prompts, targets
        
        elif dataset_name == "openorca":
            # OpenOrca: prompt为 system_prompt + question，target为 response
            for item in merged_dataset:
                system_prompt = item.get("system_prompt", "").strip()
                question = item.get("question", "").strip()
                response = item.get("response", "").strip()
                if system_prompt and question:
                    prompt = f"{system_prompt}\n{question}"
                else:
                    prompt = question or system_prompt
                if prompt and response:
                    prompts.append(prompt)
                    targets.append(response)
            return prompts, targets
        
        for item in merged_dataset:
            try:
                p, t = None, None
                options = ["A", "B", "C", "D","E","F"]
                if dataset_name == 'mmlu':
                    p = f"{item['question']}\n{format_choices(item['choices'])}\nAnswer:"
                    t = item['choices'][item['answer']]
                    
                elif dataset_name == 'arc':
                    choices = item['choices']['text']
                    labels = item['choices']['label']
                    if item['answerKey'] in labels:
                        target_idx = labels.index(item['answerKey'])
                        p = f"{item['question']}\n{format_choices(choices)}"
                        t = f"{item['answerKey']}. {choices[target_idx]}"
                    
                elif dataset_name == 'gpqa':
                    # 构造四个选项，正确答案放A，错误答案放BCD
                    choices = [
                        item.get('Correct Answer', '').strip(),
                        item.get('Incorrect Answer 1', '').strip(),
                        item.get('Incorrect Answer 2', '').strip(),
                        item.get('Incorrect Answer 3', '').strip()
                    ]
                    # 问题+选项
                    p = f"{item.get('Question', '').strip()}\n{format_choices(choices)}\n"
                    # 答案解析
                    explanation = item.get('Explanation', '').strip()
                    # 找出正确答案的选项号
                    answer_idx = 0  # 默认A为正确答案
                    answer_text = choices[answer_idx]
                    t = f"{explanation}\nAnswer: {options[answer_idx]}. {answer_text}"
                    
                elif dataset_name == 'math':
                    p = f"{item['problem']}\nSolution:"
                    t = item['solution']
                    
                elif dataset_name == 'gsm8k':
                    p = f"{item['question']}\nAnswer:"
                    t = item['answer']
                    
                elif dataset_name == 'hellaswag':
                    label_idx = int(item['label'])
                    choices = item['endings']
                    p = f"{item['ctx']}\nEndings:\n{format_choices(item['endings'])}"
                    t = f"{options[label_idx]}. {choices[label_idx]}"
                    
                elif dataset_name == 'mbpp':
                    p = f"{item['text']}\nCode:"
                    t = item['code']
                    
                elif dataset_name == 'humaneval':
                    p = item['prompt']
                    t = item['canonical_solution']
                    
                elif dataset_name == 'piqa':
                    # 构造两个选项，sol1为A，sol2为B
                    choices = [item.get('sol1', '').strip(), item.get('sol2', '').strip()]
                    p = f"{item.get('goal', '').strip()}\n{format_choices(choices)}"
                    answer_idx = int(item.get('label', 0))
                    answer_text = choices[answer_idx]
                    t = f"{options[answer_idx]}. {answer_text}"
                
                elif dataset_name == 'commonsenseqa':
                    choices = item['choices']['text']
                    labels = item['choices']['label']
                    answer_key = item['answerKey']
                    if answer_key in labels:
                        target_idx = labels.index(answer_key)
                        # 格式化为选择题
                        p = f"{item['question']}\n" + "\n".join([f"{l}. {txt}" for l, txt in zip(labels, choices)])
                        t = f"{labels[target_idx]}. {choices[target_idx]}"

                elif dataset_name == 'webquestions':
                    p = f"{item['question']}"
                    t = " || ".join(item['answers']) if item['answers'] else ""

                elif dataset_name == 'nq_open':
                    p = f"{item['question']}"
                    t = " || ".join(item['answer']) if item['answer'] else ""
                
                elif dataset_name == 'dailydialog':
                    utterances = item['utterances']
                    if len(utterances) > 1:
                        dialog_lines = []
                        for idx, utt in enumerate(utterances[:-1]):
                            role = "A" if idx % 2 == 0 else "B"
                            dialog_lines.append(f"{role}: {utt}")
                        p = "Dialog:\n" + "\n".join(dialog_lines) + "\nResponse:"
                        t = utterances[-1]

                elif dataset_name == 'e2e':
                    p = f"{item['mr']}"
                    t = item['ref']

                elif dataset_name == 'alpaca':
                    input_str = f"\nInput: {item['input']}" if item['input'] else ""
                    p = f"Instruction: {item['instruction']}{input_str}\nOutput:"
                    t = item['output']

                elif dataset_name == 'dolly':
                    context_str = f"\nContext: {item['context']}" if item['context'] else ""
                    p = f"Instruction: {item['instruction']}{context_str}\nResponse:"
                    t = item['response']

                elif dataset_name == 'spider':
                    p = f"Question: {item['question']}\nSQL:"
                    t = item['query']

                elif dataset_name == 'sharegpt':
                    convs = item['conversations']
                    if convs and len(convs) >= 2:
                        last_msg = convs[-1]
                        if last_msg['from'] == 'gpt':
                            history = []
                            for msg in convs[:-1]:
                                role = "User" if msg['from'] == 'human' else "Assistant"
                                val = msg['value']
                                history.append(f"{role}: {val}")
                            p = "\n".join(history) + "\nAssistant:"
                            t = last_msg['value']
                
                elif dataset_name == 'wikitext':
                    text = item.get('text', '').strip()
                    if len(text) > 50:
                        mid = len(text) // 2
                        p = text[:mid]
                        t = text[mid:]
                
                elif dataset_name == 'qqp':
                    p = item.get('src', '').strip()
                    t = item.get('trg', '').strip()
                
                elif dataset_name == 'wmt14':
                    # WMT14: 机器翻译任务，德译英
                    translation = item.get('translation', {})
                    de = translation.get('de', '').strip()
                    en = translation.get('en', '').strip()
                    if de and en:
                        p = de
                        t = en
                
                elif dataset_name == 'dailymail':
                    # CNN-DailyMail: 文本摘要任务
                    article = item.get('article', '').strip()
                    highlights = item.get('highlights', '').strip()
                    if article and highlights:
                        p = article
                        t = highlights
                
                elif dataset_name == 'rocstories':
                    sentences = [
                        item.get('sentence1', ''),
                        item.get('sentence2', ''),
                        item.get('sentence3', ''),
                        item.get('sentence4', ''),
                        item.get('sentence5', '')
                    ]
                    storytitle = item.get('storytitle', '').strip()
                    if all(sentences):
                        if storytitle:
                            p = f"{storytitle}\n" + " ".join(sentences[:4])
                        else:
                            p = " ".join(sentences[:4])
                        t = sentences[4]
                
                elif dataset_name == 'lm1b':
                    # LM1B (One Billion Word Benchmark): 语言建模任务
                    # 类似 WikiText，将句子分成前后两部分
                    text = item.get('text', '').strip()
                    if len(text) > 50:
                        # 按单词分割，更自然
                        words = text.split()
                        if len(words) > 6:
                            mid_word_idx = len(words) // 2
                            p = " ".join(words[:mid_word_idx])
                            t = " ".join(words[mid_word_idx:])
                
                if p and t:
                    prompts.append(p)
                    targets.append(t)
                    
            except Exception as e:
                continue  # 跳过格式错误的数据

        return prompts, targets

    except Exception as e:
        print(f"\033[1;31m处理数据集 {dataset_name} 时出错: {e}\033[0m")
        return [], []

def split_tarin_test(dataset_name,data_path,data_ratio=1.0):
    if dataset_name=='e2e':
        train_prompts, train_targets,test_prompts, test_targets=load_and_process_dataset(
            dataset_name, data_path
        )
        return train_prompts, train_targets,test_prompts, test_targets
    # --- 1. 加载并预处理所有数据 ---
    all_prompts, all_targets = load_and_process_dataset(dataset_name, data_path)
    
    if not all_prompts:
        print("错误: 未能加载任何数据，请检查路径或数据集名称。")
        return

    print(f"共加载 {len(all_prompts)} 条有效数据。")

    # --- 2. 按最大比例裁剪 ---
    total = len(all_prompts)
    max_count = int(total * data_ratio)
    if data_ratio < 1.0:
        print(f"仅使用前 {max_count} 条数据（占比 {data_ratio*100:.1f}%）参与划分。")
        all_prompts = all_prompts[:max_count]
        all_targets = all_targets[:max_count]
        total = max_count
    
    # --- 2. 执行 70/30 随机划分 ---
    # 70% 用于构建 RAG 检索库 (Train)，30% 用于测试 (Test/Validation)
    print("正在执行 70/30 随机划分...")
    
    # 组合并打乱
    combined = list(zip(all_prompts, all_targets))
    random.seed(42) # 保证可复现
    random.shuffle(combined)
    
    split_idx = int(len(combined) * 0.7)
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]
    
    train_prompts, train_targets = zip(*train_data) if train_data else ([], [])
    test_prompts, test_targets = zip(*test_data) if test_data else ([], [])
    
    # 转回 list
    train_prompts, train_targets = list(train_prompts), list(train_targets)
    test_prompts, test_targets = list(test_prompts), list(test_targets)
    return train_prompts, train_targets,test_prompts, test_targets
    
    
def process_prompt(prompt_text: str, dataset_name: str) -> str:
    """
    Prompt 处理逻辑
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'arc':
        # 这里的 instruction 要和你的 RAG Draft 格式呼应
        prefix = "Question: "
        # 强制要求 Label 开头
        suffix = "\nInstruction: Select the correct option. Start your answer with the option letter (e.g., A, B, C, D).\nAnswer:"
        return f"{prefix}{prompt_text}{suffix}"
        
    elif dataset_name == 'mmlu':
        prefix = "Question: "
        suffix = "\nAnswer:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'hellaswag':
        prefix = "Context:\n"
        suffix = ("\nInstruction: Choose the most reasonable ending from the options above. "
                  "Start your answer with the option letter (A, B, C, D) and the full ending text, "
                  "for example: D. , the man continues removing the snow on his car.\nAnswer:")
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'gpqa':
        prefix = "Question: "
        suffix = ("\nInstruction: First, give your reasoning process. Then, in the last line, output your final answer in the format: "
                  "Answer: A. 10^-4 eV (start with the option letter and full option text).")
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'piqa':
        prefix = "Question: "
        suffix = ("\nInstruction: Select the most appropriate solution. Start your answer with the option letter (A or B) and the full option text, "
                  "for example: B. Pour it into a jar\nAnswer:")
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'commonsenseqa':
        prefix = "Question: "
        suffix = ("\nInstruction: Select the most appropriate answer. Start your answer with the option letter (A, B, C, D, E) and the full option text, "
                  "for example: A. ignore\nAnswer:")
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name in ['nq_open','webquestions']:
        prefix = "Question: "
        suffix = "\nInstruction: Please answer the question with a short and direct response.\nAnswer:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'dailydialog':
        prefix = ""
        suffix = "\nInstruction: Please continue the conversation and say the next sentence as naturally as possible.\nResponse:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'e2e':
        prefix = "Meaning Representation:\n"
        suffix = "\nInstruction: Please generate a fluent and natural sentence describing the above information.\nOutput:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'wikitext':
        prefix = "Text:\n"
        suffix = "\nInstruction: Please continue the above text.\nContinuation:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'qqp':
        prefix = "Question:\n"
        suffix = "\nInstruction: Please rewrite the above question with the same meaning but different wording.\nRewritten Question:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'rocstories':
        prefix = "Story:\n"
        suffix = "\nInstruction: Please write the final sentence to complete the story.\nEnding:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'lm1b':
        prefix = "Text:\n"
        suffix = "\nInstruction: Please complete the above sentence or paragraph naturally.\nCompletion:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "banking77":
        label_list = "\n".join([f"{idx:02d}. {name}" for idx, name in enumerate(banking77_label_names)])
        prefix = (
            "Task: Intent Recognition for Banking Customer Support.\n"
            "Background: Given a user's query related to banking services, your goal is to identify the most appropriate intent category from a predefined list. "
            "Each intent represents a specific banking-related issue or request that a customer might have.\n"
            "Below are all possible intent categories (index. intent):\n"
            f"{label_list}\n\n"
            "Please read the user's input and output ONLY the most suitable intent in the format: [index]. [intent], for example: 11. card_arrival\n"
            "User input:"
        )
        return f"{prefix}{prompt_text}\nYour answer:"
    
    elif dataset_name == "commongen":
        prefix = (
            "Task: CommonGen - Generative Commonsense Reasoning.\n"
            "Background: Given a set of concepts, your goal is to generate a fluent, meaningful, and commonsense English sentence that expresses a plausible scenario involving all the given concepts.\n"
            "Concepts: "
        )
        suffix = (
            "\nInstruction: Please write one sentence that naturally and logically connects all the above concepts."
            "\nOutput:"
        )
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'wmt14':
        prefix = (
            "Task: Machine Translation (German to English).\n"
            "Background: Given a sentence in German, your goal is to translate it into fluent and accurate English.\n"
            "German sentence:\n"
        )
        suffix = "\nInstruction: Please provide the English translation.\nEnglish translation:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == 'dailymail':
        prefix = (
            "Task: Summarization (CNN/DailyMail).\n"
            "Background: Given a news article, your goal is to generate a concise and informative summary that captures the key points of the article.\n"
            "Article:\n"
        )
        suffix = "\nInstruction: Please write a summary of the above article in 1-3 sentences.\nSummary:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "bitext_customer":
        prefix = (
            "Task: Customer Support Response Generation.\n"
            "Background: Given a customer inquiry, your goal is to generate a professional, helpful, and accurate response as a customer support agent. "
            "Please address the user's request clearly and politely, following standard customer service practices.\n"
            "Customer inquiry:\n"
        )
        suffix = "\nInstruction: Please write the full response you would send to the customer.\nResponse:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "ecommerce":
        prefix = (
            "Task: E-Commerce Customer Service Response Generation.\n"
            "Background: Given a chat history between a customer and a service agent, your goal is to generate a helpful, polite, and accurate reply as the customer service agent.\n"
            "Chat history:\n"
        )
        suffix = "\nInstruction: Please write the next response as the customer service agent.\nResponse:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "multiwoz":
        prefix = (
            "Task: Task-Oriented Dialogue Response Generation (MultiWoz-2.2).\n"
            "Background: Given a user's utterance in a customer service scenario, your goal is to generate an appropriate, helpful, and contextually relevant response as the system.\n"
            "User utterance:\n"
        )
        suffix = "\nInstruction: Please write the next system response.\nSystem response:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "ms_marco":
        prefix = (
            "Task: Open-domain Question Answering (MS MARCO).\n"
            "Background: Given a question, your goal is to answer it as accurately and concisely as possible.\n"
            "Please read the question below and write a short, direct answer.\n"
        )
        suffix = "\nInstruction: Write your answer in one or two sentences.\nAnswer:"
        return f"{prefix}{prompt_text}{suffix}"
    
    elif dataset_name == "medqa":
        prefix = (
            "Task: Medical Multiple-Choice Question Answering (MedQA-USMLE).\n"
            "Background: Given a clinical scenario and several answer options, your goal is to select the most likely answer based on the information provided.\n"
            "Please read the question and options below, then choose the best answer.\n"
        )
        suffix = (
            "\nInstruction: Output ONLY the correct answer in the format: [option letter]. [option text], for example: C. Scorpion sting\nAnswer:"
        )
        return f"{prefix}{prompt_text}{suffix}"
    
    else:
        return prompt_text

def process_retrieved_kickstart_text(retrieved_kickstart_text: str, dataset_name: str) -> str:
    """
    Prompt 处理逻辑
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['arc','medqa']:
        # return "Option. "+retrieved_kickstart_text[3:]
        return retrieved_kickstart_text[3:]
    elif dataset_name == "banking77":
        return retrieved_kickstart_text[4:]
    elif dataset_name in ['nq_open','webquestions',"commongen",'ms_marco']:
        # 拆分并只取第一个答案
        if "||" in retrieved_kickstart_text:
            return retrieved_kickstart_text.split("||")[0].strip()
        else:
            return retrieved_kickstart_text.strip()
    else:
        return retrieved_kickstart_text



def load_evaluation_metrics(dataset_name: str) -> Dict[str, object]:
    """
    根据数据集类型加载所需的评估指标。
    
    参数:
        dataset_name: 数据集名称
    
    返回:
        Dict[str, object]: 指标名称到 evaluate 库指标对象的映射
    """
    print("\n[加载评估指标]")
    metric_objs = {}
    start_time = time.time()
    try:
        if dataset_name in ['nq_open', 'webquestions', 'dailydialog', 'e2e', 'rocstories', 'qqp', 'alpaca', 'dolly', 'sharegpt', 'commongen', 'wmt14', 'dailymail', 'bitext_customer', 'ecommerce', 'multiwoz', 'ms_marco', 'openorca']:
            metric_objs['rouge'] = evaluate.load('rouge')
            print("  ✓ ROUGE 加载成功")
        
        if dataset_name in ['nq_open', 'webquestions', 'dailydialog', 'e2e', 'rocstories', 'qqp', 'alpaca', 'dolly', 'sharegpt', 'commongen', 'wmt14', 'dailymail', 'bitext_customer', 'ecommerce', 'multiwoz', 'ms_marco', 'openorca']:
            metric_objs['bertscore'] = evaluate.load("bertscore")
            print("  ✓ BERTScore 加载成功")
        
        if dataset_name in ['nq_open', 'webquestions', 'dailydialog', 'e2e', 'qqp', 'alpaca', 'dolly', 'sharegpt', 'commongen', 'wmt14', 'dailymail', 'bitext_customer', 'ecommerce', 'multiwoz', 'ms_marco', 'openorca']:
            metric_objs['bleu'] = evaluate.load('bleu')
            print("  ✓ BLEU 加载成功")
            
        if dataset_name in ['nq_open', 'webquestions', 'dailydialog', 'e2e', 'qqp', 'commongen', 'wmt14', 'dailymail', 'bitext_customer', 'ecommerce', 'multiwoz', 'ms_marco', 'openorca']:
            metric_objs['meteor'] = evaluate.load('meteor')
            print("  ✓ METEOR 加载成功")
        elapsed_time = time.time() - start_time
        print(f"\033[1;32m   -> 加载评估指标必要模型总耗时: {elapsed_time:.2f}s\033[0m")
    except Exception as e:
        print(f"  ✗ 加载警告: {e}")
    
    return metric_objs





# 导出支持的数据集列表
SUPPORTED_DATASETS = [
    'mmlu', 'arc', 'gpqa', 'math', 'gsm8k', 'hellaswag', 'mbpp', 'humaneval', 'piqa',
    'nq_open', 'commonsenseqa', 'webquestions', 'dailydialog', 'e2e', 'alpaca', 'dolly', 'spider', 'sharegpt',
    'wikitext', 'qqp', 'rocstories', 'lm1b'
    'banking77','commongen','wmt14', 'dailymail',
    'bitext_customer', 'ecommerce', 'multiwoz',
    'ms_marco', 'mdeqa', 'openorca'

]


if __name__ == "__main__":
    # 测试 GPQA 数据集处理
    dataset_name = "banking77"
    base_path = "/root/autodl-tmp/datasets"  # 根据你的实际路径调整
    prompts, targets = load_and_process_dataset(dataset_name, base_path)
    print("样例 Prompt：")
    print(prompts[0])
    print("\n样例 Target：")
    print(targets[0])