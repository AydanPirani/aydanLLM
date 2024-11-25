import re
from simple_tokenizer import SimpleTokenizer, build_vocab
from bytepair_tokenizer import BytePairTokenizer
from tokens import END_TOKEN, UNKNOWN_TOKEN


def main():
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