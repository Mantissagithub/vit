import torch
import torch.nn as nn
import math
from layerNorm import LayerNorm

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))