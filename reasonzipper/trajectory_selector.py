"""Training-only projector from teacher checkpoints → existing embedding rows."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckpointProjector(nn.Module):
    """P_phi: [h_start; h_end; delta] → embedding space for cosine match to E[S]."""

    def __init__(self, hidden_size: int, emb_size: int, *, width: int = 1024):
        super().__init__()
        in_dim = hidden_size * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, emb_size),
        )

    def forward(self, h_start: torch.Tensor, h_end: torch.Tensor) -> torch.Tensor:
        delta = h_end - h_start
        x = torch.cat([h_start, h_end, delta], dim=-1)
        return self.net(x)


def cosine_logits(
    u: torch.Tensor,
    emb_rows: torch.Tensor,
    *,
    tau: float = 0.07,
) -> torch.Tensor:
    """u: [B, D], emb_rows: [K, D] → logits [B, K]."""
    u_n = F.normalize(u.float(), dim=-1)
    e_n = F.normalize(emb_rows.float(), dim=-1)
    return (u_n @ e_n.T) / max(float(tau), 1e-6)


def hard_assign(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def usage_entropy(assignments: Sequence[int], k: int) -> float:
    if not assignments:
        return 0.0
    counts = torch.zeros(k, dtype=torch.float32)
    for a in assignments:
        counts[int(a)] += 1.0
    p = counts / counts.sum().clamp_min(1.0)
    p = p[p > 0]
    return float((-(p * p.log())).sum().item() / math.log(max(k, 2)))
