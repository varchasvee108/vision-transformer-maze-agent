import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from models.layers import PatchEmbedding, PositionEmbedding
from transformers import get_scheduler
from core.config import Config


class MazeTransformer(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.patch_embd = PatchEmbedding(
            embd_dim=config.model.n_embd,
            img_size=config.input_data.image_size,
            patch_size=config.model.patch_size,
        )

        grid_size = config.input_data.image_size[0] // config.model.patch_size

        self.pos_embd = PositionEmbedding(
            num_patches=grid_size * grid_size,
            n_embd=config.model.n_embd,
            grid_size=grid_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.n_embd,
            nhead=config.model.num_heads,
            dim_feedforward=config.model.n_embd * config.model.dim_ratio,
            dropout=config.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.model.num_layers,
        )

        self.ln_f = nn.LayerNorm(config.model.n_embd)
        self.head = nn.Linear(config.model.n_embd, 4)
