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
torch>=2.2.2,<2.6; sys_platform == 'darwin' and platform_machine == 'x86_64'  # Intel macOS
torch >= 2.2.2; sys_platform != 'darwin' or platform_machine != 'x86_64'   # all chapters
jupyterlab >= 4.0          # all
tiktoken >= 0.5.1          # ch02; ch04; ch05
matplotlib >= 3.7.1        # ch04; ch06; ch07
tensorflow>=2.16.2; sys_platform == 'darwin' and platform_machine == 'x86_64'  # Intel macOS
tensorflow >= 2.18.0; sys_platform != 'darwin' or platform_machine != 'x86_64'   # ch05; ch06; ch07
tqdm >= 4.66.1             # ch05; ch07
numpy >= 1.26             # dependency of several other libraries like torch and pandas
pandas >= 2.2.1            # ch06
psutil >= 5.9.5            # ch07; already installed automatically as dependency of torch