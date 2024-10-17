from params import *
from model import *
import time
import json
from dataclasses import asdict

# load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# 前200个字符。这只是一个连续的文本文件，包含了莎士比亚的所有作品，紧密相连。
print(text[:200])

# Train and Test Split
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# 实例化一个全新的模型
params = ModelArgs()
model = Llama3(params, tokenizer).to(params.device)
# 打印模型参数数量
print(sum(p.numel() for p in model.parameters()) / 1e3, 'K parameters')
print(model)

# 训练
# 用于训练的数据加载，生成一小批输入 x 和目标 y 的数据
def get_batch(split, batch_size):
    data = train_data if split == 'train' else test_data
    # 从 0 到 len(data) - max_seq_len 随机选择 batch_size 个索引，保证每个样本都有足够的上下文，并且不超过数据长度
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,)) 
    x = torch.stack([data[i:i+params.max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+params.max_seq_len+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=5): # 训练过程中估计损失
    out = {}
    model.eval() # 进入评估模式
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # 退出评估模式
    return out

# 创建优化器
# 这不是他们使用的（llama3），但这个学习率和权重衰减对我们的小型minGemma有效。
lr_init = 1e-2
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)

# 训练轮次
max_iters = 2000

# 验证轮次
eval_interval = 100

# Warmup设置
warmup_iters = 50
warmup_factor = 1e-3 # 预热因子（初始学习率乘以该因子）

lr_final = 1e-5 # 最小学习率

# 学习率变化
def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        # 预热阶段
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        # 余弦退货阶段（最小学习率）
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 训练过程
start_time = time.time()
best_loss = float('inf')
# 启用异常检测。如果您需要进行大量调试，请取消注释这些行
torch.autograd.set_detect_anomaly(True)

for iter in range(max_iters):
    # 采样一批数据
    xb, yb = get_batch('train', params.max_batch_size)
    # 前向传播
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # 更新学习率
    scheduler.step()
    
    # 每隔一段时间评估一次训练集和验证集的损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, params.max_batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'step {iter:04d}: lr {current_lr:.6f}, train loss {losses["train"]:.4f}, val loss {losses["test"]:.4f}, time elapsed: {elapsed_time:.2f}s')
# Disable anomaly detection after the training loop
#torch.autograd.set_detect_anomaly(False)

# 保存模型
name = f'models/{model.__class__.__name__}_{time.strftime("%Y%m%d_%H%M%S")}_{losses["test"]:.4f}'
torch.save(model.state_dict(), name + '.pth')
# 将数据类对象转换为字典
params_dict = asdict(params)

# 将字典序列化为 JSON 文件
with open(f'{name}.json', 'w') as f:
    json.dump(params_dict, f)