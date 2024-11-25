import re


def build_vocab(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.?_!"()\']|--|\s)' , raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    vocab_tokens = sorted(set(preprocessed))
    vocab = {token:idx for idx, token in enumerate(vocab_tokens)}

    return vocab

class SimpleTokenizer():
    def __init__(self, vocab):
        self.str_to_idx = vocab
        self.idx_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)' , text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_idx[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.idx_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def main():
    vocab = build_vocab("data/the_verdict.txt")

    tokenizer = SimpleTokenizer(vocab)
    text = """
    "It's the last he painted, you know," Mrs. Gisburn said with pardonable pride. "The last but one," she corrected herself--"but the other doesn't count, because he destroyed it."
    """

    ids = tokenizer.encode(text)
    print(ids)
    tokens = tokenizer.decode(ids)
    print(tokens)

if __name__ == "__main__":
    main()