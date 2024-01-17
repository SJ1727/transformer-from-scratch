import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, cross_attention_kv: torch.tensor):
        super(Decoder, self).__init__()
        
        layers = []
        
        for _ in range(num_layers):
            layers.append(
                AddAndNorm(MultiHeadAttention(embed_dim, num_heads))
            )
            
            layers.append(
                AddAndNorm(MultiHeadAttention(embed_dim, num_heads, masked=True, cross_attention_kv=cross_attention_kv))
            )
            
            layers.append(
                AddAndNorm(FeedForwarLayer(embed_dim))
            )
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x