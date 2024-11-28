from attention.multihead_attention import MultiHeadAttention
from dataload.dataloader import create_dataloader
from tokenizers.bytepair_tokenizer import BytePairTokenizer
import torch
from model import Model
from config import GPT_CONFIG_124M

OUTPUT_DIM=256
VOCAB_SIZE=50257

BATCH_SIZE=8
CONTEXT_SIZE=4
STRIDE=4

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


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
    tokenizer = BytePairTokenizer()
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch = [
        torch.tensor(tokenizer.encode(txt1)),
        torch.tensor(tokenizer.encode(txt2))
        ]
    
    batch = torch.stack(batch, dim=0)
    
    print(batch)
    torch.manual_seed(123)
    model = Model(GPT_CONFIG_124M)
    
    model.eval()

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = generate_text_simple(
        model=model, 
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M.context_length
    )

    decoded = tokenizer.decode(out.squeeze(0).tolist())
    print(f"Output: {decoded}")




if __name__ == "__main__":
    main()