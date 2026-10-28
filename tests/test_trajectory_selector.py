from __future__ import annotations

import torch

from reasonzipper.trajectory_selector import (
    CheckpointProjector,
    cosine_logits,
    hard_assign,
    usage_entropy,
)


def test_projector_shapes():
    p = CheckpointProjector(8, 8, width=16)
    h0 = torch.randn(4, 8)
    h1 = torch.randn(4, 8)
    u = p(h0, h1)
    assert u.shape == (4, 8)
    emb = torch.randn(10, 8)
    logits = cosine_logits(u, emb)
    assert logits.shape == (4, 10)
    assert hard_assign(logits).shape == (4,)


def test_usage_entropy():
    assert usage_entropy([0, 0, 0], 3) < usage_entropy([0, 1, 2], 3)
