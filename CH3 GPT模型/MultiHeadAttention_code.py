# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 1
#####################################
# 一个简单的Dataset实现，使用了滑动窗口的方式将文本切分成输入和目标的序列对
class GPTDatasetV1(Dataset):
    # 需要处理的文档txt，tokenizer，模型的滑动窗口大小max_length，以及滑动窗口的步长stride
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# txt是需要处理的文档，batch_size是每个batch中包含的样本数量，max_length是模型的滑动窗口大小
# stride是滑动窗口的步长，shuffle表示是否在每个epoch开始时打乱数据
# drop_last表示如果数据集大小不能被batch_size整除，是否丢弃最后一个不完整的batch
# num_workers是加载数据时使用的子进程数。
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

#####################################
# Chapter 2
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, seq_len, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 确保d_out可以被num_heads整除
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 输出前再经过一层线性层
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask",torch.triu(torch.ones(seq_len, seq_len),diagonal=1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 计算Q、K、V整个矩阵
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将Q、K、V拆分成head*head_dim
        # view()不改变数据在内存中的存放顺序，只改变切法
        # view()只按物理结构顺序读，不允许跳着读
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim) 
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 维度转换，方便后续运算
        # transpose()不改变数据在内存中的存放数据，但是改变读取步长，导致逻辑上相邻的元素在物理地址上不再相邻
        # 因此PyTorch的.is_contiguous()返回False（即呈现非连续状态）

        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算attention scores
        # queries的shape：(batch_size, num_heads, seq_len, head_dim)
        # keys的shape：(batch_size, num_heads, seq_len, head_dim)
        # attn_scores的shape：(batch_size, num_heads, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(2, 3) 

        # 设置掩码
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # 得到masked attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 将masked attention scores标准化再归一化，得到masked attention weights
        # softmax不会改变张量的形状，此时的attn_weights的shape：(batch_size, num_heads, seq_len, seq_len)，与attn_scores一致
        # keys.shape[-1]就是head_dim
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # dropout防止过拟合
        attn_weights = self.dropout(attn_weights)

        # 将维度转换回去
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # 因为前面使用了transpose，使得数据在内存中逻辑不相邻
        # 而view()只按物理结构顺序读，不允许跳着读
        # contiguous()在内存中重新开辟新空间，拷贝数据，使得数据在物理和逻辑都相邻
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
