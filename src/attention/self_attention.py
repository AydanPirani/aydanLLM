import torch
import torch.nn as nn

def simple_attention(inputs):
    attn_scores = inputs @ inputs.T
    attn_weights = torch.softmax(attn_scores, dim=1)
    context_vectors = attn_weights @ inputs
    return context_vectors

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, kqv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=kqv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        
        context_vec = attn_weights @ values
        return context_vec
    

