"""TopK Sparse Autoencoder (Gao et al. 2024 style).

Encoder:  f = TopK( ReLU( W_enc (h - b_dec) + b_enc ) )
Decoder:  h_hat = W_dec @ f + b_dec
Decoder columns are unit-normalized after each optimizer step (standard).
ReLU-before-TopK guarantees non-negative f, which the contrastive feature
filter in fit_sae_vectors.py relies on.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.copy_(self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
            self.W_enc.copy_(self.W_dec.T.contiguous())

    def encode_pre(self, h: Tensor) -> Tensor:
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, h: Tensor) -> Tensor:
        z = self.encode_pre(h)
        topk_vals, topk_idx = torch.topk(z, k=self.k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        f = self.encode(h)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Unit-normalize each decoder column (one per SAE feature)."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)
