import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, layers: int):
        super(Encoder, self).__init__()

        self.layers = []
        
        for _ in range(layers):
            self.layers.append(
                AddAndNorm(MultiHeadAttention(embed_dim, num_heads=num_heads))
            )
            
            self.layers.append(
                AddAndNorm(FeedForwarLayer(embed_dim))
            )
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x