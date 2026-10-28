#!/usr/bin/env python3
"""Add discrete <|rXXX|> tokens to a tokenizer and verify 1:1 IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.codes import make_token_list, token_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/codes.yaml")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    model_name = args.model or cfg["model_name"]
    n = int(cfg["codes"]["n_tokens"])
    tokens = make_token_list(n)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    existing = set(tok.get_vocab().keys())
    to_add = [t for t in tokens if t not in existing]
    n_added = tok.add_special_tokens({"additional_special_tokens": to_add})
    print(f"[tokens] added={n_added} vocab_size={len(tok)}", flush=True)

    # Verify each maps to exactly one distinct ID
    ids = []
    for i, t in enumerate(tokens):
        enc = tok.encode(t, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"{t} encoded to {enc} (want single id)")
        ids.append(enc[0])
        if tok.decode([enc[0]], skip_special_tokens=False).replace(" ", "") not in (
            t,
            token_name(i),
        ):
            # some tokenizers add spaces; check round-trip via convert
            conv = tok.convert_ids_to_tokens(enc[0])
            if conv != t:
                print(f"[warn] id {enc[0]} converts to {conv!r} expected {t!r}", flush=True)
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate token ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.output_dir)
    meta = {
        "base_model": model_name,
        "n_tokens": n,
        "token_ids": {token_name(i): ids[i] for i in range(n)},
        "vocab_size": len(tok),
    }
    (args.output_dir / "reasoning_tokens.json").write_text(json.dumps(meta, indent=2))
    print(f"[tokens] saved tokenizer -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
