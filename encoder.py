import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int, num_layers: int):
        """
        Encoder section of the transformer

        Args:
            embed_dim (int): The dimension of the word embeddings
            num_heads (int): The number of heads to be used
            feed_forward_dim (int): The dimension of the hidden layer in the feed forward layers
            num_layers (int): The number of encoder layers to be applied
        """
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, feed_forward_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the encoder

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output Tensor
        """
        for block in self.encoder:
            x = block(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int):
        """
        Singular layer of the encoder

        Args:
            embed_dim (int): The dimension of the word embedding
            num_heads (int): The number of heads to be used
            feed_forward_dim (int): The dimension of the hidden layer in the feed forward layers
        """
        super(EncoderBlock, self).__init__()
        self.self_attention = AddAndNorm(MultiHeadAttention(embed_dim, num_heads=num_heads), embed_dim)
        self.feed_forward = AddAndNorm(FeedForwarLayer(embed_dim, feed_forward_dim), embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the encoder block

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x