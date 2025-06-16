import torch
import torch.nn as nn

class Feip(nn.Module):
    def __init__(self, in_channels=32, patch_size=16, embedding_dims=768, image_size=244):
        super().__init__()

        self.patch_size = patch_size
        self.embedding_dims = embedding_dims
        self.image_size = image_size
        self.in_channels = in_channels

        self.num_patches = (image_size // patch_size) ** 2 #2 domensions so

        self.convalutional_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dims,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.flatten_layer = nn.Flatten(start_dim=2, end_dim=3)

        self.cls_tkoen = nn.Parameter(torch.randn(1, 1, embedding_dims))

        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dims) 
        )

    def forward(self, x):
        batch_size = x.shape[0]

        patches = self.convalutional_layer(x)

        patches = patches.flatten(2)

        patches = patches.transpose(1, 2)

        cls_tokens = self.cls_tkoen.expand(batch_size, -1, -1)

        embeddings = torch.cat((cls_tokens, patches), dim=1)

        embeddings = embeddings + self.positional_embedding

        return embeddings

