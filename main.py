import torch
import torch.nn as nn
from feip import Feip  # Your patch embedding class
from encoderBlock import EncoderBlock # Your encoder block
from mlp import MultiLayerPerceptron

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embedding_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout=0.1,
        attention_dropout=0.0
    ):
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        
        self.patch_embedding = Feip(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dims=embedding_dims,
            image_size=image_size
        )
        
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(
                d_model=embedding_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embedding_dims)
        
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dims, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, mask=None):
        
        x = self.patch_embedding(x) 
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        x = self.norm(x)
        
        cls_token = x[:, 0]  

        cls_token = self.dropout(cls_token)
        logits = self.classification_head(cls_token)
        
        return logits
    
    def get_attention_maps(self, x):
        attention_maps = []
        
        x = self.patch_embedding(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            if hasattr(encoder_layer.attention, 'attention_scores'):
                attention_maps.append(encoder_layer.attention.attention_scores)
        
        return attention_maps
