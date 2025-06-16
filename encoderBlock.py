import torch
import torch.nn as nn
from mlp import MultiLayerPerceptron
from multiHeadAttention import MultiHeadAttentionNetwork
from residualConnection import ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.0):
        super().__init__()
        
        self.attention = MultiHeadAttentionNetwork(
            d_model_size=d_model,
            h=num_heads,
            dropout=attention_dropout
        )
        
        self.mlp = MultiLayerPerceptron(
            input_dim=d_model,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        x = self.residual1(x, lambda norm_x: self.attention(norm_x, norm_x, norm_x, mask))
        
        x = self.residual2(x, self.mlp)
        
        return x
