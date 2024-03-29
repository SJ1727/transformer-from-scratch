import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import PositionalEncoding
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int, num_layers: int, sequence_length: int, vocabulary_size: int, dropout: int=0):
        super(Transformer, self).__init__()
        
        self.positional_encodings = PositionalEncoding(embed_dim, sequence_length, dropout=dropout)
        self.encoder = TransformerEncoder(embed_dim, num_heads, feed_forward_dim, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, feed_forward_dim, num_layers)
        self.linear = nn.Linear(embed_dim, vocabulary_size)

    def forward(self, source: torch.tensor, target: torch.tensor):
        # Source shape: Batch size x Sequence length x Embedding dimension
        source = self.positional_encodings(source)
        source = self.encoder(source)
        
        # Target shape: Batch size x Sequence length x Embedding dimension
        target = self.positional_encodings(target)
        
        # Batch size x Max sequence length x Embedding dimeension
        out = self.decoder(target, source)
        
        # Batch size x Max sequence length x Vocabulary size
        out = self.linear(out)

        return out