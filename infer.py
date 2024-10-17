from params import *
from model import *
import time
import json
from dataclasses import asdict

# 加载权重
# pretrained model options:
# 2m parameters, context length = 256, trained for 500 iterations w/ batch size of 32 and no dropout: 'Llama3_2024-04-19|04-00-15'
# 2m parameters, context length = 512, trained for 1000 iterations w/ batch size 32 and dropout 0.1: 'Llama3_2024-04-19|15-18-16'
# 3m parameters, context length = 512, trained for 1300 iterations w/ batch size of 24 and dropout 0.1: 'Llama3_2024-04-19|17-21-51'
name = 'Llama3_2024-04-19|17-21-51'

# Deserialize the JSON file back to a dictionary
with open(f'models/{name}.json', 'r') as f:
    params_dict = json.load(f)

# Convert the dictionary back to a dataclass object
params = ModelArgs(**params_dict)
params.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize a blank model
model = Llama3(params, tokenizer).to(params.device)  

# here's the path to a minGemma model that i've trained with roughly 1m parameters
path = f'models/{name}.pth'

# Load the saved state dictionary
model.load_state_dict(torch.load(path)) 
# REMEMBER TO CHANGE VALUES IN params TO MATCH THE MODEL YOU'VE LOADED

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e3, 'K parameters')

# If you only plan to do inference, switch to evaluation mode
model.eval()

input_str = "JULIET:\nO Romeo, Romeo! wherefore art thou R" # the classic line

# doing everything with default values
print(model.generate(input_str))

# 现在让我们使用 memory_saver_div 来利用 KV 缓存，以便在序列长度增加时实现内存使用的线性扩展，尽管可能会导致质量下降。
# memory_saver_div 必须是 2 的幂，并用于计算注意力矩阵中查询序列长度维度的最大长度。
output = model.generate(
    input_str, 
    max_gen_len = params.max_seq_len - len(input_str), # our model doesn't have a built-in <endoftext> token so we have to specify when to stop generating
    memory_saver_div = 8, # the largest value we'll allow our query sequence length to get. makes memory consumption linear with respect to sequence length
    temperature = 0.6, # this is the default value that Llama3's official code has set
    top_p = 0.9, # this is the default value that Llama3's official code has set
    top_k = 32, # meta's code doesn't actually implement top_k selection but i've added it anyways as an alternative
)
print(output)