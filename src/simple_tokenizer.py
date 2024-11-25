import re
from tokens import END_TOKEN, UNKNOWN_TOKEN

class SimpleTokenizer():
    def __init__(self, vocab):
        self.str_to_idx = vocab
        self.idx_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)' , text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_idx else UNKNOWN_TOKEN for item in preprocessed]
        ids = [self.str_to_idx[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.idx_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text