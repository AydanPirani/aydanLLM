import tiktoken
from tokens import END_TOKEN

class BytePairTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={END_TOKEN})
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)