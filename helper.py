import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, einsum
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
        mask = torch.ones(logits.shape(0), logits.shape(1))
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
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(cross_attention_kv, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(cross_attention_kv, "b d (w n)->b d n w", n=self.num_heads)
        
        return q, k, v
    
    def _self_attention_projection(self, x: torch.tensor):
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        
        return q, k, v

    def forward(self, x: torch.tensor, cross_attention_kv: torch.tensor=None, masked: bool=False) -> torch.tensor:
        # Breaking up into multiple heads
        if cross_attention_kv is not None:
            q, k, v = self._cross_attention_projection(x, cross_attention_kv)
        else:
            q, k, v = self._self_attention_projection(x)
        
        # Pass through linear layers
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # Apply scaled dot product on all the heads
        output = self._scaled_dot_product(q, k, v, masked=masked)
        
        # Concatonating the output
        output = rearrange(output, "b d n w->b d (n w)")
        
        # Pass through output layer
        output = self.out(output)
        return output

class Residual(nn.Module):
    def __init__(self, func):
        super(Residual, self).__init__()
        self.func = func

    def forward(self, x, **kwargs):
        x = x + self.func(x, kwargs)
        return x
    
class AddAndNorm(nn.Module):
    def __init__(self, func):
        super(AddAndNorm, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm()

    def forward(self, x, **kwargs):
        x = Residual(self.func)(x, kwargs)
        x = self.norm(x)
        return x

class FeedForwarLayer(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForwarLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_num_embeddings: int, dropout: Optional[int]=0):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_num_embeddings).unsqueeze(1)
        div_term = torch.exp(-(torch.arange(0, embed_dim, 2) * np.log(10000.0) / embed_dim))
        terms = positions * div_term
        self.positional_encodings = torch.zeros(1, max_num_embeddings, embed_dim)
        self.positional_encodings[0, :, 0::2] = torch.sin(terms)
        self.positional_encodings[0, :, 1::2] = torch.cos(terms)

    def forward(self, x):
        x = x + self.positional_encodings[:x.size(0)]
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    # test = torch.Tensor([[[1, 2, 3, 4], [2, 2, 2, 2]], [[0, 0, 0, 0], [5, 6, 7, 8]]])
    # a = MultiHeadAttention(4, 2)
    # mask = torch.tensor([
    #     [1, 0],
    #     [1, 1]
    # ])
    # x = a(test, mask=None)
    # print(x)
    
    test = torch.tensor([1, 1, 1])
    func = lambda x: 2 * x
    res = AddAndNorm(func)(test)
    print(res)