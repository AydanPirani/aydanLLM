import re
from simple_tokenizer import SimpleTokenizer
from bytepair_tokenizer import BytePairTokenizer
from tokens import END_TOKEN, UNKNOWN_TOKEN

def build_vocab(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.?_!"()\']|--|\s)' , raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    vocab_tokens = sorted(set(preprocessed))
    vocab_tokens.extend([END_TOKEN, UNKNOWN_TOKEN])
    vocab = {token:idx for idx, token in enumerate(vocab_tokens)}
    return vocab

def main():
    vocab = build_vocab("data/the_verdict.txt")

    tokenizer = BytePairTokenizer()
    
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = f" {END_TOKEN} ".join((text1, text2))

    ids = tokenizer.encode(text)
    print(ids)
    tokens = tokenizer.decode(ids)
    print(tokens)

if __name__ == "__main__":
    main()