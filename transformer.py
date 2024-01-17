import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, max_num_embeddings: int, dropout: int=0):
        super(Transformer, self).__init__()
        
        self.positional_encodings = PositionalEncoding(embed_dim, max_num_embeddings, dropout=dropout)
        self.encoder = Encoder(embed_dim)
        

    def forward(self, x):

        return x