# 说明

## 文件说明

llada.py 是原生 LLaDA 的简单调用示例。主要是三个辅助函数。
mix.py 是代码的主要逻辑。
evaluation.py 是目前手动实现的评估指标计算代码。
eval_llada.py 不是我的代码，而是其他项目使用 lm_eval 进行评估的示例。
此处仅包含核心代码，模型和数据集都在别处

## lm_eval 安装

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# 如果需要处理某些特定格式（如特定数据集），可能还需要安装额外的包
pip install -e ".[multilingual,antlr]"
```

## 运行代码

```bash
python -m lm_eval --model hf \
    --model_args pretrained=your_model_path \
    --tasks my_custom_qa \
    --include_path ./custom_tasks \
    --limit 100
```