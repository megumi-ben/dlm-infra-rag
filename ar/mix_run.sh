#!/bin/bash
set -e # 如果任何命令失败，立即退出

# --- 配置 ---
#
# 1. 要运行的 Python 脚本的名称
PYTHON_SCRIPT="mix_ar.py"

TEMP=0.2
# 2. 存储日志文件的目录
LOG_DIR="mix_exp_logs/temp${TEMP}"

# 4. 每次运行测试的样本数 (为了快速迭代，可以设小一点)
NUM_EXAMPLES=3



# 批量数据集列表
DATASETS=(
    # "mmlu"
    "arc"
    # "gpqa"
    # "hellaswag"
    # "piqa"
    # "commonsenseqa"
    # "medqa"
    
    # "mbpp"
    # "humaneval"
    # "spider"
    
    # "nq_open"
    # "webquestions"
    # "ms_marco"

    # "alpaca"
    # "dolly"
    
    
    # "dailydialog"
    # "sharegpt"
    # "multiwoz"
    
    # "rocstories"
    # "dailymail"
    # "e2e"
    # "commongen"
    
    # "qqp"
    # "banking77"
    # "bitext_customer"

    # "openorca"
    # "wmt14"
    # "ecommerce"
    # "wikitext"
    # "lm1b"

    # "gsm8k"
    # "math"

)
DATA_PATH="/home/lyz/DLM/datasets"

# --- 脚本 ---

# 创建日志目录 (如果它不存在)
mkdir -p $LOG_DIR

echo "--- 启动批量实验 ---"
echo "日志将保存在: ${LOG_DIR}/"

for DATASET in "${DATASETS[@]}"; do
    echo "==> 正在运行: Dataset = ${DATASET}, Temperature = ${TEMP} (N=${NUM_EXAMPLES} 样本)..."
    LOG_FILE="${LOG_DIR}/${DATASET}_temp${TEMP}_n${NUM_EXAMPLES}.log"
    python -u $PYTHON_SCRIPT \
        --temperature $TEMP \
        --num_test_examples $NUM_EXAMPLES \
        --dataset $DATASET \
        --data_ratio 1.0 \
        --max_size 300000 \
        --data_path $DATA_PATH \
        --embed_model /backup01/DLM/model/bge-small-en-v1.5 \
        --rerank_model /backup01/DLM/model/bge-reranker-v2-m3 \
        --test_top_k 3 \
        --compare_mode rank_specific \
        --gpu_memory_utilization 0.6 \
        --max_tokens 128 \
        --top_p 0.9 \
        # > $LOG_FILE 2>&1 
        # --dynamic_gen_length \
        # --generation_mode dual_cache \
        # --threshold 0.9 \
        # --allow_draft_transfer \
        # --option_padding 2 \
      # --save_result
    echo "==> 完成. 日志已保存到: ${LOG_FILE}"
done

echo "--- 所有实验已完成 ---"