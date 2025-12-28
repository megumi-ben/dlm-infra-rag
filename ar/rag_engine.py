"""
rag_engine.py - 先进 RAG 引擎模块
- Dense Retrieval: SentenceTransformer + FAISS
- Sparse Retrieval: BM25
- Fusion: RRF (Reciprocal Rank Fusion)
- Reranking: CrossEncoder
"""

import time
import os
import pickle
from typing import List, Dict, Tuple, Union
import numpy as np
import jieba
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import itertools

class JinaReranker:
    def __init__(self, model_path: str, device: str = None):
        """
        适配 Jina Reranker V3 的包装类
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Jina Reranker from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto" # 自动根据设备选择 float16/bfloat16
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, sentences: List[List[str]], **kwargs) -> List[float]:
        """
        模拟 CrossEncoder.predict 接口
        Args:
            sentences: list of [query, doc] pairs. 
                       例如: [['q1', 'd1'], ['q1', 'd2'], ['q2', 'd3']]
        Returns:
            scores: list of float scores, 顺序与输入 sentences 严格一一对应
        """
        # Jina 的 rerank 接口是针对 (Query, List[Docs]) 设计的
        # 为了高效，我们需要探测输入中的 Query 结构
        
        all_scores = []
        
        # 1. 这种处理方式假设输入的 pairs 通常是按 query 聚类的（RAG 流程通常如此）
        # 使用 itertools.groupby 将连续相同的 query 归为一组进行批处理
        # 这样做可以极大利用 Jina 的并行推理能力
        for query, group in itertools.groupby(sentences, key=lambda x: x[0]):
            # 提取当前 query 下的所有 documents
            # group 是一个迭代器，转为 list
            pairs = list(group) 
            docs = [p[1] for p in pairs]
            
            if not docs:
                continue
                
            # 2. 调用 Jina 的原生 rerank
            # model.rerank(query, documents) 返回的是排序后的结果
            # 结构: [{'index': int, 'relevance_score': float, 'document': str}, ...]
            results = self.model.rerank(
                query, 
                docs, 
                max_query_length=512,
                top_n=len(docs), # 强制返回所有文档，不做截断，以便我们还原顺序
            )
            
            # 3. 【关键】还原顺序
            # Jina 返回的结果是按分数降序排列的，且包含原始索引 'index'
            # 我们需要按 'index' 将分数填回原来的位置，以保证和输入的 pairs 顺序一致
            current_scores = [0.0] * len(docs)
            for res in results:
                original_idx = res['index']
                score = res['relevance_score']
                current_scores[original_idx] = score
            
            all_scores.extend(current_scores)
            
        return all_scores



class QwenReranker:
    def __init__(self, model_path, device='cpu', use_fp16=False, use_flash_attn=False):
        """
        初始化 Qwen3-Reranker
        
        Args:
            model_path: 模型路径
            device: 计算设备 ('cpu' 或 'cuda')
            use_fp16: 是否使用半精度（仅 GPU 有效）
            use_flash_attn: 是否使用 flash_attention_2（需要安装 flash-attn）
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        
        # 模型加载配置
        model_kwargs = {}
        if use_fp16 and device != 'cpu':
            model_kwargs['torch_dtype'] = torch.float16
        if use_flash_attn:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device).eval()
        
        # Token IDs
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        # 预编码 prefix/suffix（避免重复编码）
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.content_max_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        
        # 中文 instruction（可根据需求切换）
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'


    def format_instruction(self, query, doc):
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def predict(self, query_doc_pairs: List[List[str]], batch_size=8):
        scores = []
        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i:i+batch_size]
            texts = [self.format_instruction(q, d) for q, d in batch]
            inputs = self.process_inputs(texts)
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_scores = batch_scores[:, 1].exp().tolist()
            scores.extend(batch_scores)
        return scores
# ==================== 全局分词函数（多进程兼容）====================
def jieba_tokenize(text: str) -> List[str]:
    """jieba 分词，用于 BM25 稀疏检索"""
    return list(jieba.cut_for_search(text))


class AdvancedRAGEngine:
    """
    先进混合检索 RAG 引擎
    - Dense Retrieval: SentenceTransformer + FAISS
    - Sparse Retrieval: BM25
    - Fusion: RRF (Reciprocal Rank Fusion)
    - Reranking: CrossEncoder
    """
    
    def __init__(self, 
                 embed_model: str = 'BAAI/bge-small-zh-v1.5',
                 rerank_model: str = 'BAAI/bge-reranker-v2-m3',
                 device: str = 'cpu'):
        """
        初始化 RAG 引擎
        
        Args:
            embed_model: 嵌入模型路径或名称
            rerank_model: 重排序模型路径或名称
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.index = None
        self.bm25 = None
        self.prompts = []
        self.targets = []
        self.is_qwen_reranker = False
        self.is_jina_reranker = False
        
        start_time = time.perf_counter()
        
        print(f"\033[1;34m1. 加载 Embedding 模型: {embed_model} ...\033[0m")
        self.retriever = SentenceTransformer(embed_model,trust_remote_code=True,device=device)
        print(f"\033[1;34m2. 加载 Reranker 模型: {rerank_model} ...\033[0m")
        # self.reranker = CrossEncoder(rerank_model,trust_remote_code=True,max_length=512, device=device)
        # 判断是否是 Qwen3-Reranker
        if 'qwen' in rerank_model.lower():
            # 根据设备决定是否启用 fp16
            use_fp16 = (device == 'cuda')
            self.reranker = QwenReranker(rerank_model, device=device, use_fp16=use_fp16)
            self.is_qwen_reranker = True
        elif 'jina' in rerank_model.lower():
            # 新增 Jina 支持
            self.reranker = JinaReranker(rerank_model, device=device)
            self.is_jina_reranker = True
        else:
            self.reranker = CrossEncoder(rerank_model, device=device)
        elapsed = time.perf_counter() - start_time
        print(f"\033[1;32m   -> RAG 加载模型总耗时: {elapsed:.2f}s\033[0m")
        # if hasattr(self.reranker, 'tokenizer') and self.reranker.tokenizer.pad_token is None:
        #     if self.reranker.tokenizer.eos_token is not None:
        #         self.reranker.tokenizer.pad_token = self.reranker.tokenizer.eos_token
        #     elif self.reranker.tokenizer.sep_token is not None:
        #         self.reranker.tokenizer.pad_token = self.reranker.tokenizer.sep_token
        #     else:
        #         self.reranker.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # # 同步设置底层 model 的 pad_token_id
        # if hasattr(self.reranker, 'model') and hasattr(self.reranker.model, 'config'):
        #     if self.reranker.model.config.pad_token_id is None:
        #         self.reranker.model.config.pad_token_id = self.reranker.tokenizer.pad_token_id
        

    def build_knowledge_base(self, prompts: List[str], targets: List[str], batch_size: int = 64, if_IndexIVFFlat: bool = True):
        """
        构建混合索引知识库
        
        Args:
            prompts: 问题/提示列表
            targets: 答案/目标列表
            batch_size: 向量编码批次大小
        """
        total_docs = len(prompts)
        print(f"\033[1;33m--- 开始构建知识库 (Total: {total_docs} docs) ---\033[0m")
        
        self.prompts = prompts
        self.targets = targets
        start_time = time.perf_counter()

        # 1. Sparse (BM25) 构建
        print(f"  [Sparse] 正在进行并行分词 (CPU核数: {cpu_count()})...")
        num_processes = max(1, min(int(cpu_count() / 2),8))
        with Pool(processes=num_processes) as pool:
            tokenized_corpus = pool.map(jieba_tokenize, prompts)
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        elapsed_bm25 = time.perf_counter() - start_time
        print(f"  [Sparse] BM25 索引构建完成。耗时：{elapsed_bm25:.2f}s")

        # 2. Dense (FAISS) 构建
        compute_satrt = time.perf_counter()
        print(f"  [Dense] 正在计算向量 (batch_size={batch_size})...")
        embeddings = self.retriever.encode(
            prompts, 
            normalize_embeddings=True, 
            show_progress_bar=True,
            batch_size=batch_size*2
        )
        compute_elapsed = time.perf_counter() - compute_satrt
        print(f"  [Sparse] 计算向量耗时：{compute_elapsed:.2f}s")
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        
        start_index = time.perf_counter()
        # 根据数据量选择索引类型
        if if_IndexIVFFlat and total_docs >= 5000000:
            nlist = int(4 * np.sqrt(total_docs))
            print(f"  [Dense] 数据量较大，使用倒排聚类索引 (IndexIVFFlat), nlist={nlist}...")
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print("  [Dense] 正在训练聚类中心...")
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(20, nlist)
        else:
            print(f"  [Dense] 数据量较小 (<100000)，使用精确索引 (IndexFlatIP)...")
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
        elapsed_index = time.perf_counter() - start_index
        print(f"  [Dense] 索引过程耗时：{elapsed_index:.2f}s")
        
        elapsed = time.perf_counter() - start_time
        print(f"\033[1;32m--- 知识库构建完成，耗时: {elapsed:.2f}s ---\033[0m")

    def save_to_disk(self, save_dir: str):
        """将知识库保存到磁盘"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"\033[1;34m--- 正在保存知识库到: {save_dir} ---\033[0m")
        
        # 1. 保存 FAISS 索引
        faiss_path = os.path.join(save_dir, "vector_index.faiss")
        faiss.write_index(self.index, faiss_path)
        
        # 2. 保存 BM25 对象和问答对
        data_path = os.path.join(save_dir, "data.pkl")
        payload = {
            "bm25": self.bm25,
            "prompts": self.prompts,
            "targets": self.targets
        }
        with open(data_path, 'wb') as f:
            pickle.dump(payload, f)
        
        # 3. 保存元信息
        meta_path = os.path.join(save_dir, "meta.txt")
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"total_docs: {len(self.prompts)}\n")
            f.write(f"index_type: {'IVFFlat' if len(self.prompts) >= 5000 else 'FlatIP'}\n")
            f.write(f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print("\033[1;32m--- 保存完成 ---\033[0m")

    def load_from_disk(self, save_dir: str):
        """从磁盘加载知识库"""
        print(f"\033[1;34m--- 正在从 {save_dir} 加载知识库 ---\033[0m")
        
        faiss_path = os.path.join(save_dir, "vector_index.faiss")
        data_path = os.path.join(save_dir, "data.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"索引文件不完整，请检查路径: {save_dir}")

        # 1. 加载 FAISS
        self.index = faiss.read_index(faiss_path)
        
        # 2. 加载 BM25 和问答对
        with open(data_path, 'rb') as f:
            payload = pickle.load(f)
            self.bm25 = payload["bm25"]
            self.prompts = payload["prompts"]
            self.targets = payload["targets"]
            
        print(f"\033[1;32m--- 加载完成 (包含 {len(self.prompts)} 条问答对) ---\033[0m")

    def _rrf_fusion(self, rank_lists: List[List[int]], k: int = 60) -> List[int]:
        """RRF (Reciprocal Rank Fusion) 融合多路召回结果"""
        rrf_score = {}
        for rank_list in rank_lists:
            for rank, doc_idx in enumerate(rank_list):
                if doc_idx not in rrf_score:
                    rrf_score[doc_idx] = 0
                rrf_score[doc_idx] += 1 / (k + rank + 1)
        return sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)


    
    def batch_search(self, queries: List[str], top_k_retrieve: int = 30, top_n_rerank: int = 3,
                     use_fusion: bool = False, use_rerank: bool = True, batch_size: int = 32):
        """
        批量混合检索 + 重排序
        
        Args:
            queries: 查询文本列表
            top_k_retrieve: 每一路的召回数量
            top_n_rerank: 重排序后返回数量
            use_fusion: 是否使用 RRF 融合
            use_rerank: 是否使用重排序器
            batch_size: 向量编码和重排序的批次大小
            
        Returns:
            结果列表的列表，每个查询对应一个结果列表
        """
        if self.index is None or self.bm25 is None:
            raise ValueError("索引未构建或未加载！请先调用 build_knowledge_base 或 load_from_disk")
        
        num_queries = len(queries)
        final_results = [[] for _ in range(num_queries)]
        
        # 1. Dense Retrieval (Batch)
        # 批量编码查询向量
        query_vecs = self.retriever.encode(
            queries, 
            batch_size=batch_size, 
            normalize_embeddings=True, 
            show_progress_bar=False
        ).astype('float32')
        
        search_start = time.perf_counter()
        # FAISS 支持矩阵查询：传入 (n_queries, dim) 返回 (n_queries, top_k)
        D_all, I_all = self.index.search(query_vecs, top_k_retrieve)
        search_elapsed = time.perf_counter() - search_start
        
        # 2. Sparse Retrieval (Optional) & Fusion Preparation
        all_candidate_indices = [] # 存储每个 query 的最终候选索引列表
        
        for i in range(num_queries):
            dense_indices = I_all[i].tolist()
            # 过滤掉 FAISS 可能返回的 -1 (当 top_k > 数据库总量时)
            dense_indices = [idx for idx in dense_indices if idx >= 0]
            
            if use_fusion:
                # 单条进行 BM25 (BM25 库本身不支持批量，但速度极快)
                tokenized_query = jieba_tokenize(queries[i])
                bm25_scores = self.bm25.get_scores(tokenized_query)
                # 获取 sparse top_k
                sparse_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieve].tolist()
                
                # RRF 融合
                fused = self._rrf_fusion([dense_indices, sparse_indices])
                candidates = fused[:top_k_retrieve]
            else:
                candidates = dense_indices
            
            # 过滤无效索引（越界检查）
            valid_indices = [idx for idx in candidates if 0 <= idx < len(self.prompts)]
            all_candidate_indices.append(valid_indices)

        # 3. Reranking (Batch)
        if use_rerank:
            # 准备所有的 (Query, Doc) 对
            all_pairs = []
            pair_counts = [] # 记录每个 query 有多少个 candidate，用于后续切分
            
            for i in range(num_queries):
                candidates = all_candidate_indices[i]
                query_text = queries[i]
                pairs = [[query_text, self.prompts[idx]] for idx in candidates]
                all_pairs.extend(pairs)
                pair_counts.append(len(pairs))
            
            if not all_pairs:
                return final_results,search_elapsed # 没有任何候选
                
            # 批量预测分数
            # 分流调用不同重排序模型
            if self.is_qwen_reranker or self.is_jina_reranker:
                all_scores = self.reranker.predict(all_pairs, batch_size=batch_size)
            else:
                all_scores = self.reranker.predict(all_pairs, batch_size=batch_size, show_progress_bar=False)
            
            # 将分数映射回对应的 query
            cursor = 0
            for i in range(num_queries):
                count = pair_counts[i]
                if count == 0:
                    continue
                    
                query_scores = all_scores[cursor : cursor + count]
                candidates = all_candidate_indices[i]
                
                # 组装当前 query 的结果
                query_results = []
                for j, score in enumerate(query_scores):
                    idx = candidates[j]
                    query_results.append({
                        "prompt": self.prompts[idx],
                        "target": self.targets[idx],
                        "score": float(score),
                        "index": idx
                    })
                
                # 排序并截断
                query_results.sort(key=lambda x: x['score'], reverse=True)
                final_results[i] = query_results[:top_n_rerank]
                
                cursor += count
        else:
            # 不使用 Rerank，直接基于现有顺序（RRF 或 Dense 顺序）返回
            for i in range(num_queries):
                candidates = all_candidate_indices[i]
                query_results = []
                # 注意：如果不 rerank，这里的分数处理会比较粗糙
                # 如果是 dense only, 可以回溯 D_all 的分数，如果是 fusion，分数意义不大
                for idx in candidates:
                    score = 0.0
                    # 尝试找回 dense score 作为参考
                    if idx in I_all[i]:
                         # 找到 idx 在 dense 结果中的位置
                        pos = np.where(I_all[i] == idx)[0][0]
                        score = float(D_all[i][pos])
                        
                    query_results.append({
                        "prompt": self.prompts[idx],
                        "target": self.targets[idx],
                        "score": score,
                        "index": idx
                    })
                # 已经是排好序的（RRF 或 Dense 序），直接截断
                final_results[i] = query_results[:top_n_rerank]
        return final_results,search_elapsed