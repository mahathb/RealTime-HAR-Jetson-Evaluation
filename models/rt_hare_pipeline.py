import torch
import torch.nn as nn

from models.baselines import RGBResNet50Extractor


class DummyIMFE(nn.Module):
    """Temporary IMFE stub until the real IMFE (Issue #5) is merged.

    Accepts a raw video tensor in [B, C, T, H, W] format (matching the
    expected IMFE input convention) and produces a per-frame feature
    sequence of shape [B, T, out_features] via global average pooling
    followed by a linear projection.

    Args:
        out_features (int): Output feature dimension per frame (default 512).

    Input:
        x (Tensor): Shape [B, C, T, H, W]

    Output:
        Tensor: Shape [B, T, out_features]
    """

    def __init__(self, out_features: int = 512):
        super(DummyIMFE, self).__init__()
        # Projects the 3-channel average to out_features dimensions
        self.fc = nn.Linear(3, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        # Global spatial + temporal average → [B, C]
        x_flat = x.mean(dim=[2, 3, 4])                # [B, C]
        # Project to feature dim and expand temporal axis
        return self.fc(x_flat).unsqueeze(1)            # [B, 1, out_features]


class RTHAREPipeline(nn.Module):
    """Master RT-HARE Integration Pipeline.

    Connects three stages as described in the paper
    (Wang et al., 2024, Section IV):

        Stage 1 — RGB Stream:
            RGBResNet50Extractor  →  [B, T, 2048]

        Stage 2 — Motion Stream (IMFE or DummyIMFE):
            motion_extractor      →  [B, T, motion_dim]

        Stage 3 — Temporal Reasoning (LSTR):
            Concatenated features →  predictions

    Args:
        lstr_model (nn.Module):  Temporal reasoning head (e.g., LSTR).
                                  Receives fused [B, T, 2048 + motion_dim]
                                  features and returns action predictions.
        motion_extractor (nn.Module, optional):
                                  IMFE or compatible module. Defaults to
                                  DummyIMFE(out_features=512) for stub testing.

    Inputs:
        rgb_tensor    (Tensor): [B, T, C, H, W]
        motion_tensor (Tensor): [B, C, T, H, W]  — IMFE input convention

    Output:
        predictions (Tensor): Output of lstr_model; shape depends on LSTR head.
    """

    def __init__(self, lstr_model: nn.Module, motion_extractor: nn.Module = None):
        super(RTHAREPipeline, self).__init__()
        self.rgb_extractor = RGBResNet50Extractor()
        self.motion_extractor = (
            motion_extractor if motion_extractor is not None else DummyIMFE()
        )
        self.lstr = lstr_model

    def forward(
        self, rgb_tensor: torch.Tensor, motion_tensor: torch.Tensor
    ) -> torch.Tensor:
        # ── Stage 1: RGB spatial features ──────────────────────────────────
        rgb_feats = self.rgb_extractor(rgb_tensor)         # [B, T, 2048]

        # ── Stage 2: Motion features (IMFE or stub) ─────────────────────────
        motion_feats = self.motion_extractor(motion_tensor)  # [B, T, motion_dim]

        # Align temporal dimension: broadcast stub output (T=1) to match T
        if motion_feats.shape[1] != rgb_feats.shape[1]:
            motion_feats = motion_feats.expand(
                -1, rgb_feats.shape[1], -1
            )                                              # [B, T, motion_dim]

        # ── Stage 3: Fuse & predict ─────────────────────────────────────────
        combined_feats = torch.cat(
            [rgb_feats, motion_feats], dim=-1
        )                                                  # [B, T, 2048 + motion_dim]

        predictions = self.lstr(combined_feats)
        return predictions
