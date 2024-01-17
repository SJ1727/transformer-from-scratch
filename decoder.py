import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super(Decoder, self).__init__()
        
        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, cross_attention_kv):
        for block in self.decoder:
            x = block(x, cross_attention_kv)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(embed_dim, num_heads=num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForwarLayer(embed_dim)

    def forward(self, x, cross_attention_kv):
        x = AddAndNorm(self.self_attention)(x)
        x = AddAndNorm(self.cross_attention)(x, cross_attention_kv=cross_attention_kv, masked=True)
        x = AddAndNorm(self.feed_forward)(x)
        return x