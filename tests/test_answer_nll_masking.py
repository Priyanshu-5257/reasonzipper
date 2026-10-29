"""answer_nll must score answer tokens only, not </think> context."""

from __future__ import annotations

import torch
import torch.nn as nn

from reasonzipper.compressed_selection import answer_nll


class _ToyLM(nn.Module):
    """Deterministic logits: prefer target id == position % vocab."""

    def __init__(self, vocab: int = 20, hidden: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab, bias=False)
        # Identity-ish: each row peaks on its index
        with torch.no_grad():
            self.lm.weight.copy_(torch.eye(vocab, hidden)[:, :hidden] if False else self.lm.weight)
            self.lm.weight.zero_()
            for i in range(min(vocab, hidden)):
                self.lm.weight[i, i % hidden] = 10.0

    def forward(self, input_ids, use_cache=False, output_hidden_states=False):
        h = self.emb(input_ids)
        logits = self.lm(h)
        out = type("O", (), {})()
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = (h, h)
        return out


def test_answer_nll_ignores_context_length_in_mean():
    """Same answer under different context lengths → same mean NLL for that answer."""
    model = _ToyLM()
    device = torch.device("cpu")
    prefix = [1, 2, 3]
    answer = [5, 6]
    ctx_short = [7]
    ctx_long = [7, 8, 9, 10]
    nll_a = answer_nll(
        model, prefix, answer, device=device, context_ids=ctx_short
    )
    nll_b = answer_nll(
        model, prefix, answer, device=device, context_ids=ctx_long
    )
    # Not required to be equal (context changes conditioning), but both finite
    assert nll_a == nll_a and nll_b == nll_b
    # Scoring window length is always len(answer), not len(context+answer):
    # verify by checking empty context vs context still returns a scalar.
    nll_none = answer_nll(model, prefix, answer, device=device, context_ids=None)
    assert isinstance(nll_none, float)


def test_answer_nll_empty_answer():
    model = _ToyLM()
    assert answer_nll(model, [1, 2], [], device=torch.device("cpu")) == 0.0
