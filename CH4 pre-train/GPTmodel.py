# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.


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
    

#####################################
# Chapter 3
#####################################
class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        # 避免除以0程序崩溃
        self.eps = 1e-5 
        # 缩放参数，初始为1，需要训练
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 偏移参数，初始为0，需要训练
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 算平均值和方差，使用有偏估计计算方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 将归一化的结果缩放偏移，使结果更灵活
        return self.scale * norm_x + self.shift
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],               # 输入特征维度
            d_out=cfg["emb_dim"],              # 输出特征维度
            seq_len=cfg["context_length"],     # 上下文长度
            dropout=cfg["drop_rate"],          # Dropout 比例
            num_heads=cfg["n_heads"],          # 注意力头的数量
            qkv_bias=cfg["qkv_bias"]           # 查询、键和值的偏置
        ) 
        self.ff = FeedForward(cfg)  # 前馈神经网络模块
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一归一化层
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二归一化层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差连接的 Dropout

    def forward(self, x):
        # 对注意力模块的快捷连接
        shortcut = x
        x = self.norm1(x)  # 应用第一归一化层
        x = self.att(x)  # 通过多头注意力模块，形状为 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        # 对前馈网络模块的残差连接
        shortcut = x
        x = self.norm2(x)  # 应用第二归一化层
        x = self.ff(x)  # 通过前馈神经网络模块
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 权重共享
        #self.out_head.weight = self.tok_emb.weight
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # embedding层
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # transformer层
        x = self.trf_blocks(x)

        # output层
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
           logits = model(idx_cond)
        # 只保留最后一个时间步的输出
        logits = logits[:, -1, :]
        # 经过softmax，计算概率
        probas = torch.softmax(logits, dim=-1)
        # 找到概率最大的token的token ID
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 将这个token ID加在列表中
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)