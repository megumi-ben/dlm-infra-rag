# dream

## 文件说明

chat_dream.py：简单的与dream模型的对话示例。
code_dream.py：使用dream模型对数据集进行测试评估的代码。
model文件夹下是dream相关代码（fast-dllm改进版），model_origin文件夹下面是原生dream代码。
model/init.py：初始化model包，便于模块导入。
model/configuration_dream.py：定义模型的配置参数和配置类。
model/generation_utils_block.py：实现分块生成相关的辅助工具函数。
model/generation_utils.py：实现文本生成的通用辅助工具函数。
model/modeling_dream.py：定义和实现Dream模型的结构与前向传播等核心逻辑。
model/tokenization_dream.py：实现文本分词和编码相关的工具和

mix.py 是我的项目（使用rag技术生成草稿draft热启动）和fast-dllm的结合代码，只不过这里只是适配了llada，而没有适配dream