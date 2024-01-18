from helper import PositionalEncoding
from transformer import Transformer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

def visualize_positional_encodings():
    ENCODING_DIM = 1000
    NUMBER_OF_ENCODINGS = 500
    encoding = PositionalEncoding(ENCODING_DIM, NUMBER_OF_ENCODINGS, dropout=0.1)
    encoding_tensors = torch.zeros(1, NUMBER_OF_ENCODINGS, ENCODING_DIM)
    encoding_tensors = encoding(encoding_tensors)

    plt.imshow(encoding_tensors.squeeze(0))
    plt.show()

def testing_transformer():
    EMBED_DIM = 512
    NUMBER_OF_HEADS = 8
    LAYERS = 6
    VOCAB_SIZE = 10000
    FEED_FORWARD_DIM = 2048
    DROPOUT = 0.1
    SEQ_LEN = 1000
    
    source = torch.randn(3, 1000, 512)
    target = torch.rand(3, 1000, 512)
    
    test_transformer = Transformer(EMBED_DIM, NUMBER_OF_HEADS, FEED_FORWARD_DIM, LAYERS, SEQ_LEN, VOCAB_SIZE, dropout=DROPOUT)
    out = test_transformer(source, target)
    
    print(out.shape)

if __name__ == "__main__":
    #visualize_positional_encodings()
    testing_transformer()