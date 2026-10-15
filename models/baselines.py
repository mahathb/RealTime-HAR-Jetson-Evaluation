import torch
import torch.nn as nn
import torchvision.models as models


class RGBResNet50Extractor(nn.Module):
    """RGB-Only Baseline Feature Extractor using pretrained ResNet50.

    Replaces the final FC classification layer with an identity to extract
    raw 2048-dimensional spatial feature vectors per frame.

    Args:
        out_dim (int): Output feature dimension. Must match ResNet50 pool
                       output (2048). Kept as a parameter for clarity.

    Input:
        x (Tensor): Shape [B, T, C, H, W]
                    B = Batch size, T = Frames, C = Channels (3),
                    H = Height, W = Width

    Output:
        feats (Tensor): Shape [B, T, 2048]
    """

    def __init__(self, out_dim: int = 2048):
        super(RGBResNet50Extractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final FC classification layer — keep everything up to
        # AdaptiveAvgPool2d to produce 2048-dim spatial feature vectors.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # Flatten batch and time dimensions for parallel frame processing
        x_reshaped = x.view(b * t, c, h, w)          # [B*T, C, H, W]

        feats = self.backbone(x_reshaped)              # [B*T, 2048, 1, 1]
        feats = feats.view(b, t, self.out_dim)         # [B, T, 2048]
        return feats
