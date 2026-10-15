"""Shared sequence encoders (point-wise representation, no instance normalization).

ReformerEncoder mirrors TSLib `models/Reformer.py` (DataEmbedding + ReformerLayer
LSH attention + EncoderLayer/Encoder) with the task head removed, so the
The Reformer baseline uses the vendored TSLib encoder layers.
"""
import torch.nn as nn

from ..vendor.tslib.layers.Embed import DataEmbedding
from ..vendor.tslib.layers.SelfAttention_Family import ReformerLayer
from ..vendor.tslib.layers.Transformer_EncDec import Encoder, EncoderLayer


class ReformerEncoder(nn.Module):
    """x [B, L, n_in] -> h [B, L, d_model].

    Embedding = TokenEmbedding (value) + sinusoidal PositionalEmbedding; no time
    features (x_mark=None). Any L works: TSLib ReformerLayer.fit_length zero-pads
    the sequence to a multiple of bucket_size * 2 and crops the output back to L.
    """

    def __init__(self, n_in, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1,
                 bucket_size=4, n_hashes=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.embedding = DataEmbedding(n_in, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, d_model, n_heads, bucket_size=bucket_size, n_hashes=n_hashes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        h = self.embedding(x, None)
        h, _ = self.encoder(h, attn_mask=None)
        return h
