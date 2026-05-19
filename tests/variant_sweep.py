#!/usr/bin/env python3
"""
Attention-variant training sweep for OpenMythos.

Trains several attention configurations on *identical* TinyStories batches —
same seed, same data order, one shared token buffer — and reports train/eval
loss and throughput side by side, so any delta reflects the attention
mechanism rather than data or init noise. The MoE router z-loss and the ACT
ponder cost are included in the objective (return_aux=True), i.e. this is the
real training loss, and the aux-loss-free router-bias step runs each step.

Default sweep (4 runs): gqa / signed / gated with cross-loop attention on,
plus gqa with cross-loop attention off (the cross-loop ablation).

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python tests/variant_sweep.py
    # quick smoke test:
    PYTHONPATH=. python tests/variant_sweep.py --steps 8 --train-tokens 300000 \
        --eval-tokens 40000 --dim 64 --n-heads 4 --loops 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from open_mythos import MythosConfig, OpenMythos
from tests.small_benchmark import (
    PackedLMDataset,
    count_params,
    evaluate,
    fmt_count,
    load_text_ds,
)

# Named sweep presets — each entry is (variant name, config overrides). Every
# variant in a preset trains on identical batches, so a delta reflects the
# swept field alone. Select with --sweep.
SWEEPS = {
    # E/I attention isolation: full-MHA baseline, excitation-inhibition and its
    # additive control (isolates the subtraction), the gated variant, and
    # signed with sigmoid score maps.
    "ei": [
        ("mha", dict(attn_type="mha")),
        ("signed", dict(attn_type="signed")),
        ("signed-plus", dict(attn_type="signed-plus")),
        ("gated", dict(attn_type="gated")),
        ("signed-sigmoid", dict(attn_type="signed-sigmoid")),
    ],
    # sigmoid attention: plain sigmoid vs the softmax baseline, and whether
    # E/I rescues or worsens it (signed-sigmoid)
    "sigmoid": [
        ("mha", dict(attn_type="mha")),
        ("sigmoid", dict(attn_type="sigmoid")),
        ("signed", dict(attn_type="signed")),
        ("signed-sigmoid", dict(attn_type="signed-sigmoid")),
    ],
    # multi-domain comparison: baseline, mass-freedom, negation-via-two-maps,
    # negation-in-one-map — run per domain to test domain-dependence.
    "domains": [
        ("mha", dict(attn_type="mha")),
        ("sigmoid", dict(attn_type="sigmoid")),
        ("signed", dict(attn_type="signed")),
        ("tanh", dict(attn_type="tanh")),
    ],
    # lighter 3-run attention-mechanism check
    "attention": [
        ("mha", dict(attn_type="mha")),
        ("signed", dict(attn_type="signed")),
        ("gated", dict(attn_type="gated")),
    ],
}


def build_cfg(vocab_size: int, args: argparse.Namespace, **overrides) -> MythosConfig:
    """A single shared 'fast sweep' config; overrides set the swept fields."""
    base = dict(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=2,
        max_seq_len=args.seq_len,
        max_loop_iters=args.loops,
        prelude_layers=2,
        coda_layers=2,
        attn_type="gqa",
        # MLA fields — must be valid even when the variant does not use MLA
        kv_lora_rank=128,
        q_lora_rank=192,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=args.dim,
        lora_rank=8,
        rope_theta=10000.0,
        dropout=0.0,
    )
    base.update(overrides)
    return MythosConfig(**base)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay >=2-D matrices only; exempt 1-D params (norms, biases, LTI gains)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup -> cosine decay."""
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * r))


def train_variant(
    name: str,
    cfg: MythosConfig,
    train_ds: PackedLMDataset,
    eval_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train one attention variant; return its metrics and loss curve."""
    torch.manual_seed(args.seed)
    model = OpenMythos(cfg).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(
        param_groups(model, 0.1), lr=args.lr, betas=(0.9, 0.95)
    )

    # Re-seed *after* model init so every variant sees the identical batch
    # order regardless of how many RNG draws its init consumed.
    torch.manual_seed(args.seed)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    print(f"\n=== {name}  ({fmt_count(n_params)} params) ===", flush=True)
    curve: list[tuple[int, float, float | None]] = []
    tok_total, t_total = 0, 0.0
    data_iter = iter(loader)
    model.train()
    loss_val = float("nan")
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = get_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp:
            logits, aux = model(x, return_aux=True)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
            ) + aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.step_router_bias()  # aux-loss-free MoE load balancing
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_total += x.numel()
        t_total += dt
        loss_val = loss.item()

        if step == 1 or step % args.log_every == 0:
            ev = None
            if args.eval_every and step % args.eval_every == 0:
                ev = evaluate(
                    model, eval_loader, device, cfg.vocab_size, args.eval_batches
                )
                model.train()
            curve.append((step, loss_val, ev))
            line = f"  [{name}] step {step:>5}/{args.steps}  loss {loss_val:.4f}"
            if ev is not None:
                line += f"  eval {ev:.4f}"
            line += f"  {x.numel() / dt:,.0f} tok/s"
            print(line, flush=True)

    final_eval = evaluate(model, eval_loader, device, cfg.vocab_size)
    print(
        f"  [{name}] done — final train {loss_val:.4f}  final eval {final_eval:.4f}"
        f"  avg {tok_total / max(1e-9, t_total):,.0f} tok/s  wall {t_total:.0f}s",
        flush=True,
    )
    # save the trained model so the variant can be instrumented afterwards
    # (e.g. inspecting a signed model's excitatory vs inhibitory maps)
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt = os.path.join(args.ckpt_dir, f"{args.sweep}_{name}.pt")
        torch.save({"cfg": cfg, "model": model.state_dict(), "name": name}, ckpt)
        print(f"  [{name}] checkpoint → {ckpt}", flush=True)

    return dict(
        name=name,
        attn_type=cfg.attn_type,
        params=n_params,
        final_train=loss_val,
        final_eval=final_eval,
        tok_per_sec=tok_total / max(1e-9, t_total),
        wall=t_total,
        curve=curve,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--sweep", default="ei", choices=sorted(SWEEPS))
    p.add_argument(
        "--ckpt-dir",
        default="variant_sweep_ckpts",
        help="dir to save each variant's trained model ('' disables)",
    )
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-tokens", type=int, default=40_000_000)
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--dataset-config", default="")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument(
        "--text-field",
        default="text",
        help="dataset text field; comma-separated fields are concatenated",
    )
    p.add_argument(
        "--eval-split",
        default="validation",
        help="eval split name, or '@train-skip' to hold out from the train stream",
    )
    p.add_argument(
        "--eval-skip",
        type=int,
        default=100_000,
        help="examples to skip ahead for '@train-skip' eval (keeps it disjoint)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="variant_sweep_results.json")
    return p.parse_args()


def prep_stream(raw, text_field: str):
    """
    Return (stream, field) with a single usable text field. A comma-separated
    text_field concatenates those fields into a synthesised 'text' field — so
    QA-style datasets (e.g. question + answer) work without a flat text column.
    """
    fields = [f.strip() for f in text_field.split(",")]
    if len(fields) == 1:
        return raw, fields[0]
    return raw.map(lambda s: {"text": "\n".join(str(s[f]) for f in fields)}), "text"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(
        f"[setup] device={device}  dim={args.dim}  loops={args.loops}  "
        f"seq_len={args.seq_len}  batch={args.batch_size}  steps={args.steps}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    print(f"[setup] tokenizer={args.tokenizer}  vocab={vocab_size:,}", flush=True)

    # One shared token buffer — materialized once, reused by every variant.
    raw_train, tfield = prep_stream(
        load_text_ds(args.dataset, args.dataset_config, "train"), args.text_field
    )
    train_ds = PackedLMDataset(
        raw_train, tokenizer, args.seq_len, args.train_tokens, tfield
    )
    if args.eval_split == "@train-skip":
        # dataset has no held-out split — hold eval out by skipping ahead in
        # the train stream so the eval tokens are disjoint from training.
        raw_eval = load_text_ds(args.dataset, args.dataset_config, "train").skip(
            args.eval_skip
        )
    else:
        raw_eval = load_text_ds(args.dataset, args.dataset_config, args.eval_split)
    raw_eval, _ = prep_stream(raw_eval, args.text_field)
    eval_ds = PackedLMDataset(
        raw_eval, tokenizer, args.seq_len, args.eval_tokens, tfield
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[setup] dataset={args.dataset}  eval_split={args.eval_split}", flush=True)
    print(
        f"[setup] train tokens={train_ds.data.numel():,} pairs={len(train_ds)}  "
        f"eval tokens={eval_ds.data.numel():,} pairs={len(eval_ds)}",
        flush=True,
    )

    sweep = SWEEPS[args.sweep]
    print(
        f"[setup] sweep='{args.sweep}'  variants={[n for n, _ in sweep]}", flush=True
    )
    results: list[dict] = []
    t_start = time.perf_counter()
    for name, overrides in sweep:
        cfg = build_cfg(vocab_size, args, **overrides)
        results.append(train_variant(name, cfg, train_ds, eval_loader, args, device))
        # write incrementally so a mid-sweep crash still leaves partial data
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    bar = "=" * 78
    print(f"\n{bar}\nVariant sweep — {time.perf_counter() - t_start:.0f}s total\n{bar}")
    head = (
        f"  {'variant':<14} {'params':>9} {'final train':>12} "
        f"{'final eval':>11} {'tok/s':>10} {'wall(s)':>9}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    best = min(results, key=lambda r: r["final_eval"])
    for r in results:
        mark = "  <- best eval" if r is best else ""
        print(
            f"  {r['name']:<14} {fmt_count(r['params']):>9} "
            f"{r['final_train']:>12.4f} {r['final_eval']:>11.4f} "
            f"{r['tok_per_sec']:>10,.0f} {r['wall']:>9.0f}{mark}"
        )
    print(f"\n[done] results + loss curves written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
