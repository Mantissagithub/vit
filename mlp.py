import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, mlp_ratio = 4.0, dropout= 0.0):
        super().__init__()
        
        hidden_dim = int(input_dim * mlp_ratio)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x