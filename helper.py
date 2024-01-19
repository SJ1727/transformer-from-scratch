import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Optional

torch.manual_seed(0)

class MultiHeadAttention(nn.Module):    
    def __init__(self, embed_dim: int, num_heads: int=1):
        """
        Implementation of the multihead attention system presented in the paper
        https://arxiv.org/abs/1706.03762
        Args:
            embed_dim (int): The dimenstion of the word embeddings
            num_heads (int): The number of heads to be used
        """

        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise Exception("Embed dimension must be divisable by the number of heads")
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

    def _mask_logits(self, logits: torch.tensor) -> torch.tensor:
        """
        Applies a mask to the logits

        Args:
            logits (torch.tensor): The logits to be masked

        Returns:
            torch.tensor: masked logits
        """
        
        mask = torch.ones(logits.size(2), logits.size(3))
        mask = torch.tril(mask, diagonal=0)

        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return masked_logits

    def _scaled_dot_product(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, masked: Optional[bool]=False) -> torch.tensor:
        """
        Applies scaled dot product using query, key, and value with an optional mask

        Args:
            q (torch.tensor): query
            k (torch.tensor): key
            v (torch.tensor): value
            mask (Optional[bool], optional): Should a mask be applied

        Returns:
            torch.tensor: Applies scaled dot product
        """
        attention_logits = torch.matmul(q, torch.transpose(k, -2, -1))
        attention_logits *= 1/np.sqrt(self.head_dim)
        
        if masked:
            attention_logits = self._mask_logits(attention_logits)

        attention = torch.softmax(attention_logits, dim=-1)
        
        ## Causes grad_fn=<UnsafeViewBackward0>, might be a problem later idk
        values = torch.matmul(attention, v)
        
        return values

    def _cross_attention_projection(self, x: torch.tensor, cross_attention_kv: torch.tensor) -> tuple[torch.tensor]:
        """
        Creates the query, key, and value vectors where the key and value vectors come from the other input tensor

        Args:
            x (torch.tensor): Tensor to obtain query vectors
            cross_attention_kv (torch.tensor): Tensor to obtain key and value vectors

        Returns:
            tuple[torch.tensor]: query, key, value
        """
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(cross_attention_kv, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(cross_attention_kv, "b d (w n)->b d n w", n=self.num_heads)
        
        return q, k, v
    
    def _self_attention_projection(self, x: torch.tensor) -> tuple[torch.tensor]:
        """
        Creates the query, key, and value vectors from a tensor

        Args:
            x (torch.tensor): Tensor to get the vectors from

        Returns:
            tuple[torch.tensor]: query, key, value
        """
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        
        return q, k, v

    def forward(self, x: torch.tensor, cross_attention_kv: Optional[torch.tensor]=None, masked: Optional[bool]=False) -> torch.tensor:
        """
        Forward pass of the attention layer

        Args:
            x (torch.tensor): input tensor
            cross_attention_kv (Optional[torch.tensor], optional): Optional tensor which can be used to get the key and values vectors in the case of cross attention. Defaults to None.
            masked (Optional[bool], optional): Can apply a mask to the tensor. Defaults to False.

        Returns:
            torch.tensor: Output tensor after attention layer
        """
        # Breaking up into multiple heads
        if cross_attention_kv is not None:
            q, k, v = self._cross_attention_projection(x, cross_attention_kv)
        else:
            q, k, v = self._self_attention_projection(x)
        
        # Pass through linear layers
        # Batch size x Sequence length x Number of heads x Head dim
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # Apply scaled dot product on all the heads
        output = self._scaled_dot_product(q, k, v, masked=masked)
        
        # Concatonating the output
        # Batch size x Sequence length x embedding dim
        output = rearrange(output, "b d n w->b d (n w)")
        
        # Pass through output layer
        output = self.out(output)
        return output

class Residual(nn.Module):
    def __init__(self, func):
        """
        Applies residual connection after function

        Args:
            func (Function): Function before residual connection
        """
        super(Residual, self).__init__()
        self.func = func

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Forward pass of residual layer

        Args:
            x (torch.tensor): Tensor before funcion and residual

        Returns:
            torch.tensor: Output tensor
        """
        x = x + self.func(x, **kwargs)
        return x
    
class AddAndNorm(nn.Module):
    def __init__(self, func, embed_dim: int):
        """
        Applies residual connection and layer normilisation

        Args:
            func (Function): function to be applied before residual and normilisation
            embed_dim (int): Dimention of the input tensor
        """
        super(AddAndNorm, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Forward pass of the Add and Norm layer

        Args:
            x (torch.tensor): Tensor before function, residual, and normilisation

        Returns:
            torch.tensor: Ouput tensor
        """
        x = Residual(self.func)(x, **kwargs)
        x = self.norm(x)
        return x

class FeedForwarLayer(nn.Module):
    def __init__(self, embed_dim: int, feed_forward_dim: int):
        """
        Feed forward layer

        Args:
            embed_dim (int): dimension of the input
            feed_forward_dim (int): dimension of the hidden layer
        """
        super(FeedForwarLayer, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the feed forward layer

        Args:
            x (torch.tensor): Tensor before the feed forward layer

        Returns:
            torch.tensor: Output tensor
        """
        x = self.layers(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, sequence_length: int, dropout: Optional[int]=0.1):
        """
        Applies positional encodding on sequence of vectors

        Args:
            embed_dim (int): size of the embedding vectors
            sequence_length (int): maximum sequence length
            dropout (Optional[int], optional): Dropout probability. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(-(torch.arange(0, embed_dim, 2) * np.log(10000.0) / embed_dim))
        terms = positions * div_term
        self.positional_encodings = torch.zeros(1, sequence_length, embed_dim)
        self.positional_encodings[0, :, 0::2] = torch.sin(terms)
        self.positional_encodings[0, :, 1::2] = torch.cos(terms)

    def forward(self, x):
        """
        Forward pass of the positional encodding

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = x + self.positional_encodings[:, :x.size(1)]
        x = self.dropout(x)
        return x