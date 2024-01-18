import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int, num_layers: int, sequence_length: int, vocabulary_size: int, dropout: int=0):
        super(Transformer, self).__init__()
        
        self.positional_encodings = PositionalEncoding(embed_dim, sequence_length, dropout=dropout)
        self.encoder = Encoder(embed_dim, num_heads, feed_forward_dim, num_layers)
        self.decoder = Decoder(embed_dim, num_heads, feed_forward_dim, num_layers)
        self.linear = nn.Linear(embed_dim, vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, source: torch.tensor, target: torch.tensor):
        source = self.positional_encodings(source)
        source = self.encoder(source)
        
        target = self.positional_encodings(target)
        out = self.decoder(target, source)
        
        out = self.linear(out)
        out = self.softmax(out)

        return out