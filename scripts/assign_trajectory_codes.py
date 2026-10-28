#!/usr/bin/env python3
"""Train P_phi on teacher checkpoints and assign hard existing-token codes.

Loss: maximize cosine(P_phi(h), E[target]) where target is the NN of h_end among S
as a bootstrapped alignment target (then hard-assign from P_phi). Also a mild
usage entropy bonus via batch-level KL toward uniform (optional).
Writes SFT JSONL for B-2 distillation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.existing_vocab import (
    assert_original_vocabulary,
    build_candidate_token_ids,
    ids_to_think_body,
)
from reasonzipper.schemas import format_gold_answer
from reasonzipper.trajectory_selector import (
    CheckpointProjector,
    cosine_logits,
    hard_assign,
    usage_entropy,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="trajectory SFT jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--candidate-k", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--codes-per-block", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    index = json.loads((args.checkpoints / "index.json").read_text())
    items = index["items"]
    hidden_size = int(index["meta"]["hidden_size"])

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map=args.device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    assert_original_vocabulary(base, tok, label="assign")
    emb = base.get_input_embeddings().weight.detach()
    # Keep embedding rows on CPU fp16; move slices to device as needed
    candidates = build_candidate_token_ids(tok, k=args.candidate_k, seed=args.seed)
    E_s = emb[candidates].to(torch.float32).to(args.device)
    emb_size = int(E_s.shape[1])

    # Load all blocks
    h_starts = []
    h_ends = []
    owners = []  # (item_idx, block_idx)
    for ii, it in enumerate(items):
        shard = torch.load(args.checkpoints / it["path"], map_location="cpu")
        hs = shard["h_start"].float()
        for bi in range(int(shard["n_blocks"])):
            h_starts.append(hs)
            h_ends.append(shard["h_ends"][bi].float())
            owners.append((ii, bi))
    H0 = torch.stack(h_starts, dim=0)
    H1 = torch.stack(h_ends, dim=0)
    print(f"[assign] blocks={len(owners)} hidden={hidden_size} K={len(candidates)}", flush=True)

    # Bootstrapped targets: NN of h_end in E[S]
    with torch.no_grad():
        # process in chunks
        targets = []
        bs = 256
        for i in range(0, H1.shape[0], bs):
            chunk = H1[i : i + bs].to(args.device)
            # project h_end to emb dim via truncated/pad if needed — use linear map init
            # For NN in emb space, use first emb_size dims of h_end (Qwen hidden==emb usually)
            if chunk.shape[-1] != emb_size:
                if chunk.shape[-1] > emb_size:
                    chunk = chunk[..., :emb_size]
                else:
                    pad = torch.zeros(chunk.shape[0], emb_size - chunk.shape[-1], device=chunk.device)
                    chunk = torch.cat([chunk, pad], dim=-1)
            logits = cosine_logits(chunk, E_s, tau=args.tau)
            targets.append(hard_assign(logits).cpu())
        targets = torch.cat(targets, dim=0)

    projector = CheckpointProjector(hidden_size, emb_size).to(args.device)
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    n = H0.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            h0 = H0[idx].to(args.device)
            h1 = H1[idx].to(args.device)
            y = targets[idx].to(args.device)
            u = projector(h0, h1)
            logits = cosine_logits(u, E_s, tau=args.tau)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        print(f"[assign] epoch {ep+1}/{args.epochs} loss={total/n:.4f}", flush=True)

    # Hard assign with trained projector
    projector.eval()
    assigned_local: list[list[int]] = [[] for _ in items]
    all_assign: list[int] = []
    with torch.no_grad():
        for i in range(0, n, bs):
            h0 = H0[i : i + bs].to(args.device)
            h1 = H1[i : i + bs].to(args.device)
            u = projector(h0, h1)
            logits = cosine_logits(u, E_s, tau=args.tau)
            # top codes_per_block
            topk = min(args.codes_per_block, logits.shape[-1])
            vals, inds = logits.topk(topk, dim=-1)
            for j in range(inds.shape[0]):
                ii, bi = owners[i + j]
                local = [int(candidates[int(x)]) for x in inds[j].tolist()]
                assigned_local[ii].extend(local)
                all_assign.extend([int(x) for x in inds[j].tolist()])

    ent = usage_entropy(all_assign, len(candidates))
    print(f"[assign] usage_entropy={ent:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for ii, it in enumerate(items):
            ids = assigned_local[ii][:8] or [int(candidates[0])]
            body = ids_to_think_body(tok, ids)
            gold = it["first_error_line"]
            completion = f"<think>\n{body}\n</think>\n{format_gold_answer(gold)}"
            row = {
                "id": it["id"],
                "prompt": it["prompt"],
                "completion": completion,
                "gold_answer": it["gold_answer"],
                "first_error_line": gold,
                "code_token_ids": ids,
                "n_codes": len(ids),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    torch.save(
        {
            "projector": projector.state_dict(),
            "candidates": candidates,
            "hidden_size": hidden_size,
            "emb_size": emb_size,
            "usage_entropy": ent,
        },
        args.output.with_suffix(".selector.pt"),
    )
    meta = {
        "n": n_out,
        "candidate_k": args.candidate_k,
        "usage_entropy": ent,
        "epochs": args.epochs,
        "codes_per_block": args.codes_per_block,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[assign] wrote {n_out} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
