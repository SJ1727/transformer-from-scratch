import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import AddAndNorm, MultiHeadAttention, FeedForwarLayer

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int, num_layers: int):
        """
        Decoder Section of the transformer

        Args:
            embed_dim (int): The dimension of the word embeddings
            num_heads (int): The number of heads to be used
            feed_forward_dim (int): The dimension of the feed forward layers hidden layer
            num_layers (int): The number of decoder layers
        """
        super(TransformerDecoder, self).__init__()
        
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, feed_forward_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.tensor, cross_attention_kv: torch.tensor) -> torch.tensor:
        """
        Forward pass of the decoder

        Args:
            x (torch.tensor): Input tensor
            cross_attention_kv (torch.tensor): Tensor to be used in cross attention

        Returns:
            torch.tensor: Output tensor
        """
        for block in self.decoder:
            x = block(x, cross_attention_kv)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int):
        """
        Signular layer of a decoder

        Args:
            embed_dim (int): Word embedding dimension
            num_heads (int): Number of heads to be used
            feed_forward_dim (int): The dimension of the hidden layer in the feed forward layers
        """
        super(TransformerDecoderBlock, self).__init__()
        
        self.self_attention = AddAndNorm(MultiHeadAttention(embed_dim, num_heads=num_heads), embed_dim)
        self.cross_attention = AddAndNorm(MultiHeadAttention(embed_dim, num_heads=num_heads), embed_dim)
        self.feed_forward = AddAndNorm(FeedForwarLayer(embed_dim, feed_forward_dim), embed_dim)

    def forward(self, x: torch.tensor, cross_attention_kv: torch.tensor) -> torch.tensor:
        """
        Forward pass of the decoder block

        Args:
            x (torch.tensor): Input tensor
            cross_attention_kv (torch.tensor): Tesnor to be used in cross attention

        Returns:
            torch.tensor: Output tensor
        """
        x = self.self_attention(x, masked=True)
        x = self.cross_attention(x, cross_attention_kv=cross_attention_kv)
        x = self.feed_forward(x)
        return x
    