import torch
import pytest
from models.model import MazeTransformer
from models.layers import PatchEmbedding, PositionEmbedding
from core.config import Config
from pathlib import Path


def test_patch_embedding():
    config = Config.load(Path("config/base.toml"))
    B, C, H, W = 2, 3, 132, 132
    patch_size = 12
    embd_dim = 256

    expected_grid_size = H // patch_size * W // patch_size

    patch_embd = PatchEmbedding(
        img_size=H, patch_size=patch_size, in_channels=C, embd_dim=embd_dim
    )
    x = torch.randn(B, C, H, W)
    out = patch_embd(x)

    assert out.shape == (B, expected_grid_size, embd_dim), (
        f"Expected output shape to be {B, expected_grid_size, embd_dim}, got {out.shape}"
    )


def test_pos_embd():
    B, T, C = 2, 121, 256

    grid_size = 11

    pos_layer = PositionEmbedding(num_patches=T, n_embd=C, grid_size=grid_size)
    x = torch.randn(B, T, C)
    out = pos_layer(x)

    assert out.shape == (B, T, C), (
        f"Expected output shape to be {B, T, C}, got {out.shape}"
    )

    assert not torch.allclose(out[0, 0], out[0, 1]), (
        "Expected output to be different for different patches"
    )

    print("Position embedding test passed")
