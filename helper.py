import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, einsum
import numpy as np
from typing import Optional

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(Attention, self).__init__()
        if embed_dim % num_heads != 0:
            raise Exception("Embed dimension must be divisable by the number of heads")
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

    def _mask_logits(self, logits: torch.tensor, mask: torch.IntTensor) -> torch.tensor:
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return masked_logits

    def _scaled_dot_product(self, q, k, v, mask: Optional[torch.IntTensor]=None) -> torch.tensor:
        attention_logits = torch.matmul(q, torch.transpose(k, -2, -1))
        attention_logits *= 1/np.sqrt(self.head_dim)
        
        if mask is not None:
            attention_logits = self._mask_logits(attention_logits, mask=mask)

        attention = torch.softmax(attention_logits, dim=-1)
        
        ## Causes grad_fn=<UnsafeViewBackward0>, might be a problem later idk
        values = torch.matmul(attention, v)
        
        return values

    def forward(self, x: torch.tensor, mask: Optional[torch.IntTensor]=False) -> torch.tensor:
        # Breaking up into multiple heads
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        
        # Pass through linear layers
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # Apply scaled dot product on all the heads
        output = self._scaled_dot_product(q, k, v, mask=mask)
        
        # Concatonating the output
        output = rearrange(output, "b d n w->b d (n w)")
        
        # Pass through output layer
        output = self.out(output)
        return output

if __name__ == "__main__":
    test = torch.Tensor([[[1, 2, 3, 4], [2, 2, 2, 2]], [[1, 1, 1, 1], [5, 6, 7, 8]]])
    a = Attention(4, 2)
    mask = torch.tensor([
        [1, 0],
        [1, 1]
    ])
    x = a(test, mask=mask)
    print(x)