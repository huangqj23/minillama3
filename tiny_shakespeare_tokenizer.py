import pickle
import os


class SimpleTokenizer:
    def __init__(self, stoi, merges):
        self.stoi = stoi
        self.merges = merges
        self.itos = {i: s for s, i in stoi.items()}  # inverse mapping for decoding
        
        self.vocab_len = len(stoi) + len(merges)
        
    def encode(self, text):
        # 将文本转换为标记 ID 的列表，对于未知字符使用空格
        tokens = [self.stoi.get(c, self.stoi[' ']) for c in text]
        
        # 执行合并，允许嵌套合并的可能性
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges: # self.merges是一个字典:{(token1, token2): merged_token, (token3, token4): merged_token2}
                # 用合并的标记替换当前的对
                merged_token = self.merges[pair]
                tokens[i] = merged_token
                del tokens[i + 1]
                
                # 返回以处理可能的嵌套合并
                if i > 0:
                    i -= 1
            else:
                i += 1
        return tokens
    
    def decode(self, tokens):
        def expand_token(token):
            # 基本情况：如果token是直接映射，则返回其字符
            if token in self.itos:
                return self.itos[token]
            # 递归情况：如果该token是合并标记，则展开其组成部分
            elif token in self.merges.values():
                pair = next(key for key, value in self.merges.items() if value == token)
                return ''.join(expand_token(t) for t in pair)
            # 未知token的后备方案
            else:
                return ''
        # 解码列表中的每个token，递归处理嵌套合并
        return ''.join(expand_token(token) for token in tokens)
            
def load_tokenizer_data(size: int):
    file_name = f'./tokenizers/tiny_shakespeare_tokenizer_{size}.model'
    with open(file_name, 'rb') as f:
        tokenizer_data = pickle.load(f)
    return tokenizer_data

def get_tokenizer(size: int):
    tokenizer_data = load_tokenizer_data(size)
    loaded_stoi = tokenizer_data['stoi']
    loaded_merges = tokenizer_data['merges']
    return SimpleTokenizer(loaded_stoi, loaded_merges)
            