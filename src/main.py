from attention.multihead_attention import MultiHeadAttention
from dataload.dataloader import create_dataloader
import torch, torch.nn as nn

OUTPUT_DIM=256
VOCAB_SIZE=50257

BATCH_SIZE=8
CONTEXT_SIZE=4
STRIDE=4

def main():
    # with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()

    # token_embedding_layer = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    # pos_embedding_layer = nn.Embedding(CONTEXT_SIZE, OUTPUT_DIM)
    
    # dataloader = create_dataloader(raw_text, batch_size=BATCH_SIZE, max_length=CONTEXT_SIZE, stride=STRIDE, shuffle=False)
    # data_iter = iter(dataloader)
    
    # inputs, targets = next(data_iter)
    # token_embeddings = token_embedding_layer(inputs)
    # pos_embeddings = pos_embedding_layer(torch.arange(CONTEXT_SIZE))
    # input_embeddings = token_embeddings + pos_embeddings

    # print(input_embeddings.shape)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64], 
        [0.22, 0.58, 0.33], 
        [0.77, 0.25, 0.10], 
        [0.05, 0.80, 0.55]
    ])

    d_in = 3
    d_out = 2
    torch.manual_seed(123)

    batch = torch.stack((inputs, inputs))
    context_length = batch.shape[1]

    attention = MultiHeadAttention(d_in, d_out, context_length, 0.0, 2)
    rv = attention(batch)
    print(rv.shape)
    print(rv)


if __name__ == "__main__":
    main()