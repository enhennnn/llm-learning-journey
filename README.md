## License & Attribution

This repository is a derivative work based on [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)  
by Sebastian Raschka, which is licensed under the Apache License 2.0.

All code in this repository is also provided under the Apache License 2.0.  
See the [LICENSE](LICENSE) file for the full license text.

Modifications and additional code by [enhennnn].


# 我的大模型学习之旅

这个仓库用来记录我的 LLM 学习笔记和代码。

## 环境要求

- **Python**：3.10.20

## 依赖包

使用的 package 定义在 `requirement.txt` 中，具体如下：

```text
# PyTorch GPU 版本（支持 RTX 5060，需 CUDA 12.8）
torch>=2.7.0
torchvision>=0.22.0
torchaudio>=2.7.0

# 项目其他依赖
jupyterlab>=4.0
tiktoken>=0.5.1
matplotlib>=3.7.1
tensorflow>=2.18.0
tqdm>=4.66.1
numpy>=1.26
pandas>=2.2.1
psutil>=5.9.5
```

## 项目目录结构

```bash
.
├── ch1_处理文本数据/
│   ├── 01_简单分词法——按单词分token
│   ├── 02_创建token词表，将token和token_ID对应
│   ├── 03_增加特殊的token
│   ├── 04_Byte_pair_encoding(BPE)
│   ├── 05_使用滑动窗口对数据采样
│   ├── 06_token_embeddings
│   ├── 07_增加位置信息
│   └── 08_总结
│
├── ch2_注意力机制/
│   ├── 01_简单的self-attention
│   ├── 02_带可训练参数的self-attention
│   ├── 03_自注意力模块
│   ├── 04_因果注意力_causal_attention
│   ├── 05_因果注意力模块
│   └── 06_多头注意力
│
├── ch3_GPT模型/
│   ├── 01_GPT结构
│   ├── 02_归一化
│   ├── 03_激活函数GELU与前向传播
│   ├── 04_残差链接
│   ├── 05_组合成Transformer模块
│   ├── 06_GPT_model
│   └── 07_生成文字
│
├── ch4_pre-train/
│   ├── 01_回顾之前的模型
│   ├── 02_损失函数：交叉熵和困惑度
│   ├── 03_计算训练集和验证集的损失
│   ├── 04_训练模型
│   ├── 05_增加decoding的随机性
│   ├── 06_保存与导入模型参数
│   └── 07_从OpenAI中导入参数
│
├── ch5_Classification_Fine-tuning/
│   ├── 01_准备数据集
│   ├── 02_创建Dataloader
│   ├── 03_使用预训练权重初始化模型
│   ├── 04_修改输出层为分类层
│   ├── 05_分类的正确率和损失loss
│   ├── 06_微调——监督训练
│   ├── 07_使用模型作为垃圾邮件识别器
│   └── 08_模型保存和再次使用
│
└── ch6_Instruction_Fine-tuning/
    ├── 01_准备数据集
    ├── 02_创建Dataset和collate_fn
    ├── 03_创建Dataloader
    ├── 04_加载预训练的大模型
    ├── 05_指令微调
    └── 06_保存模型回复
```