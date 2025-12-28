# 说明

rag_engine.py 是rag技术相关，负责知识库构建、检索，可以切换嵌入模型和重排序模型。
data_processor.py 是数据集的处理相关，加载、处理相关逻辑。
evaluation.py 是评估指标计算相关逻辑。
mix.py 是DLM模型和rag技术检索草稿热启动相关逻辑，也是主流程入口，核心是initialize_with_kickstart.
chat_ar.py 是使用vllm框架和AR模型进行对话的示例代码。

我的项目逻辑是DLM推理加速，思想是rag技术检索草稿作为draft作为DLM推理的热启动，从而减少推理步数。其中baseline使用qwen和llama进行对比，这两个ar模型也将和rag技术（就是普通的rag技术，放到prompt中，不mask）结合，作为对比存在。
目前要实现的就是AR模型结合rag技术以及评估指标的计算（主流程实现），请给出实现代码。可以参考mix.py以及chat_ar.py