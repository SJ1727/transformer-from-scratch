from helper import PositionalEncoding
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

def visualize_positional_encodings():
    ENCODING_DIM = 512
    NUMBER_OF_ENCODINGS = 250
    encoding = PositionalEncoding(ENCODING_DIM, NUMBER_OF_ENCODINGS)
    encoding_tensors = torch.zeros(NUMBER_OF_ENCODINGS, ENCODING_DIM)
    encoding_tensors = encoding(encoding_tensors)

    plt.imshow(encoding_tensors)
    plt.show()

if __name__ == "__main__":
    visualize_positional_encodings()