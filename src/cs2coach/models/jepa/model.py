"""Action-conditioned JEPA over CS2 entity tokens.

Context encoder E(s_t) -> token latents. Target encoder = EMA copy of E.
Predictor P(E(s_t), a, Δ) -> predicted target latents of s_{t+Δ}.
Loss is in latent space (no reconstruction of raw positions), with a
VICReg-style variance/covariance term on pooled context latents to stop
the encoder collapsing to a constant.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs2coach.features.state_tensor import ACTION_DIM, N_TOKENS, TOKEN_DIM

DELTAS = (8, 16, 32, 64)  # in samples at 8 Hz: 1 s, 2 s, 4 s, 8 s


class Encoder(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.GELU(), nn.Linear(d, d))
        # Slot embedding distinguishes CT players / T players / global, not individual players.
        self.kind = nn.Embedding(3, d)
        self.register_buffer("kinds", torch.tensor([0] * 5 + [1] * 5 + [2]), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, TOKEN_DIM) -> (B, N, d)
        h = self.inp(x) + self.kind(self.kinds)
        return self.norm(self.tf(h))


def pool(z: torch.Tensor) -> torch.Tensor:
    return z.mean(1)


class Predictor(nn.Module):
    def __init__(self, d: int = 256, layers: int = 3, heads: int = 4):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(ACTION_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.dt = nn.Embedding(len(DELTAS), d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, d)

    def forward(self, z: torch.Tensor, a: torch.Tensor, dt_idx: torch.Tensor) -> torch.Tensor:
        cond = (self.act(a) + self.dt(dt_idx)).unsqueeze(1)          # (B, 1, d)
        h = self.tf(torch.cat([z + cond, cond], dim=1))[:, :N_TOKENS]
        return self.out(h)


class JEPA(nn.Module):
    def __init__(self, d: int = 256, enc_layers: int = 4, pred_layers: int = 3, ema: float = 0.996):
        super().__init__()
        self.d = d
        self.encoder = Encoder(d, enc_layers)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = Predictor(d, pred_layers)
        self.ema = ema

    @torch.no_grad()
    def update_target(self) -> None:
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(self.ema).add_(p.detach(), alpha=1 - self.ema)

    def loss(self, s0, a, dt_idx, s1, var_w: float = 1.0, cov_w: float = 0.04):
        z0 = self.encoder(s0)
        with torch.no_grad():
            z1 = self.target(s1)
        pred = self.predictor(z0, a, dt_idx)
        l_pred = F.smooth_l1_loss(pred, z1)

        p = pool(z0)
        p = p - p.mean(0)
        std = torch.sqrt(p.var(0) + 1e-4)
        l_var = F.relu(1.0 - std).mean()
        cov = (p.T @ p) / max(p.shape[0] - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        l_cov = off.pow(2).sum() / p.shape[1]
        total = l_pred + var_w * l_var + cov_w * l_cov
        return total, dict(pred=l_pred.item(), var=l_var.item(), cov=l_cov.item(), std=std.mean().item())

    @torch.no_grad()
    def embed(self, s: torch.Tensor) -> torch.Tensor:
        """Pooled target-space latent: what probes and clustering consume."""
        return pool(self.target(s))

    @torch.no_grad()
    def predict(self, s0: torch.Tensor, a: torch.Tensor, dt_idx: torch.Tensor) -> torch.Tensor:
        return pool(self.predictor(self.encoder(s0), a, dt_idx))


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
