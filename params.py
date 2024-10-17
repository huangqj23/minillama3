import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional # 用于表示一个变量可以是某种特定类型，也可以是 None。
from tiny_shakespeare_tokenizer import *

tokenizer = get_tokenizer(size=512) # 尺寸选项为128、256、512和1024

'''
dataclass 是 Python 3.7 引入的一个装饰器，用于简化类的定义，特别是那些主要用于存储数据的类。
使用 dataclass 可以自动生成一些常用的特殊方法，比如 __init__()、__repr__()、__eq__() 等，从而减少样板代码的编写。
'''
@dataclass
class ModelArgs:
    dim: int = 128 # 4096
    n_layers: int = 12 # 32
    n_heads: int = 4 # 32
    n_kv_heads: Optional[int] = 1 # None
    vocab_size: int = tokenizer.vocab_len # -1
    multiple_of: int = 256 # 使 SwiGLU 隐藏层大小成为大于 2 的幂的倍数
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000 # 500000
    max_batch_size: int = 24
    max_seq_len: int = 512 # 8192，但他们在运行推理时的最大块大小是2048。
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate: float = 0.1