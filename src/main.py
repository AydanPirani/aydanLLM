from dataload.dataloader import create_dataloader
import torch, torch.nn as nn

OUTPUT_DIM=256
VOCAB_SIZE=50257

BATCH_SIZE=8
CONTEXT_SIZE=4
STRIDE=4

def main():
    with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    token_embedding_layer = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    pos_embedding_layer = nn.Embedding(CONTEXT_SIZE, OUTPUT_DIM)
    
    dataloader = create_dataloader(raw_text, batch_size=BATCH_SIZE, max_length=CONTEXT_SIZE, stride=STRIDE, shuffle=False)
    data_iter = iter(dataloader)
    
    inputs, targets = next(data_iter)
    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(CONTEXT_SIZE))
    input_embeddings = token_embeddings + pos_embeddings

    print(input_embeddings.shape)


if __name__ == "__main__":
    main()