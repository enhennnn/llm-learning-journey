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