import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForwarLayer(embed_dim)

    def forward(self, x):
        x = AddAndNorm(self.self_attention)(x)
        x = AddAndNorm(FeedForwarLayer)(x)
        return x