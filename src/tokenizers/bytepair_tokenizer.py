import tiktoken
from tokenizers.tokens import END_TOKEN

class BytePairTokenizer:
    def __init__(self):
        # self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.tokenizer = tiktoken.encoding_for_model("gpt-2")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={END_TOKEN})
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)