import re

def build_vocab(tokens):
    vocab_tokens = sorted(set(tokens))
    vocab = {token:idx for idx, token in enumerate(vocab_tokens)}
    return vocab

def tokenize(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.?_!"()\']|--|\s)' , raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed

def main():
    tokens = tokenize("data/the_verdict.txt")
    vocab = build_vocab(tokens)

if __name__ == "__main__":
    main()