#!/usr/bin/env python3
"""
Issue #3 — RGB Baseline & RT-HARE Pipeline Integration: Verification Script
===========================================================================
Validates the full forward pass of RTHAREPipeline using synthetic tensors
to confirm that all tensor shapes propagate correctly end-to-end.

Expected output shape: [B, T, num_classes]  →  [2, 16, 50]

Usage:
    # Run from the repository root:
    python scripts/verify_issue3.py
"""

import sys
import os

# Ensure repo root is on the path so `models` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.rt_hare_pipeline import RTHAREPipeline


class DummyLSTR(torch.nn.Module):
    """Minimal LSTR stub: single linear layer over the fused feature dim."""

    def __init__(self, in_dim: int = 2560, num_classes: int = 50):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]  →  [B, T, num_classes]
        return self.fc(x)


def main():
    print("=" * 60)
    print("  RT-HARE Issue #3 — Pipeline Integration Verification")
    print("=" * 60)

    # ── Configuration ────────────────────────────────────────────
    BATCH      = 2
    FRAMES     = 16
    CHANNELS   = 3
    HEIGHT     = 224
    WIDTH      = 224
    NUM_CLASS  = 50

    # RGB input:    [B, T, C, H, W]   (frame-first, standard dataloader format)
    dummy_rgb    = torch.randn(BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH)

    # Motion input: [B, C, T, H, W]   (channel-first, IMFE convention)
    dummy_motion = torch.randn(BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH)

    print(f"\n  Input  RGB tensor shape:    {list(dummy_rgb.shape)}")
    print(f"  Input Motion tensor shape:  {list(dummy_motion.shape)}")

    # ── Build pipeline ───────────────────────────────────────────
    # fused_dim = ResNet50(2048) + DummyIMFE(512) = 2560
    lstr_stub = DummyLSTR(in_dim=2560, num_classes=NUM_CLASS)
    pipeline  = RTHAREPipeline(lstr_model=lstr_stub)
    pipeline.eval()

    # ── Forward pass ─────────────────────────────────────────────
    print("\n  Running forward pass ...", end=" ", flush=True)
    with torch.no_grad():
        out = pipeline(dummy_rgb, dummy_motion)
    print("done.")

    # ── Shape assertion ──────────────────────────────────────────
    expected = (BATCH, FRAMES, NUM_CLASS)
    assert out.shape == torch.Size(expected), (
        f"Shape mismatch! Got {tuple(out.shape)}, expected {expected}"
    )

    print(f"\n[PASS] Output tensor shape: {list(out.shape)}")
    print(f"       Expected:             {list(expected)}")
    print("\n  Pipeline verification executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
