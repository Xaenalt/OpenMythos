#!/usr/bin/env python3
"""
Attention-mechanism instrumentation for the E/I sweep checkpoints.

Loads the trained variant_sweep checkpoints (variant_sweep_ckpts/) and probes
whether sigmoid attention and signed (excitation/inhibition) attention are
*substitutes* — i.e. doing the same job — or genuinely distinct.

Probe 1 — inhibition gain. The learned per-head γ = sigmoid(inhib_gain) in the
`signed` vs `signed-sigmoid` models, per layer. γ→0 would mean the model
abandoned E/I once sigmoid was present.

Probe 2 — attention map structure. Runs a batch of TinyStories validation text
through each model, recomputes every attention layer's weight map from the
captured layer input, and measures, on long-context queries (second half of
the sequence):
  - softmax maps: effective # keys attended (perplexity of the row).
  - sigmoid maps: total attention mass Σσ per query (sigmoid has no
    normalisation, so this is "how much it attends") + effective # keys of the
    normalised row (shape).
  - signed models: exc/inhib overlap — histogram intersection of the two
    normalised maps per query. High → inhibition suppresses the same keys
    excitation attends (a contrast/sharpening op); low → inhibition targets
    different keys (distinct negative information).

    PYTHONPATH=. python tests/analyze_attention.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from open_mythos import OpenMythos
from open_mythos.main import MHAttention, SignedAttention, apply_rope
from tests.small_benchmark import PackedLMDataset, load_text_ds


def load_model(path: str, device: torch.device):
    """Rebuild a model from a sweep checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = OpenMythos(ck["cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["cfg"], ck["name"]


# ---------------------------------------------------------------------------
# Probe 1 — learned inhibition gain
# ---------------------------------------------------------------------------


def probe_inhibition_gain(model, name: str) -> None:
    print(f"  [{name}]  γ = sigmoid(inhib_gain), per attention layer:")
    for mod_name, m in model.named_modules():
        if isinstance(m, SignedAttention) and hasattr(m, "inhib_gain"):
            g = torch.sigmoid(m.inhib_gain.detach().float().cpu())
            print(
                f"    {mod_name:24s} mean {g.mean():.3f}  "
                f"range [{g.min():.3f}, {g.max():.3f}]"
            )


# ---------------------------------------------------------------------------
# Probe 2 — attention map structure
# ---------------------------------------------------------------------------


def _causal_mask(T: int, device, dtype) -> torch.Tensor:
    return torch.triu(
        torch.full((1, 1, T, T), float("-inf"), device=device, dtype=dtype), 1
    )


@torch.no_grad()
def attention_maps(module, x, freqs_cis) -> dict:
    """Recompute an attention module's weight map(s) from its captured input."""
    B, T, _ = x.shape
    H, d = module.n_heads, module.head_dim
    mask = _causal_mask(T, x.device, x.dtype)
    scale = d**-0.5

    def proj(w):  # (B,T,dim) -> (B,H,T,d) with RoPE
        return apply_rope(w(x).view(B, T, H, d), freqs_cis).transpose(1, 2)

    sigmoid = getattr(module, "score_fn", "softmax") == "sigmoid"
    if isinstance(module, SignedAttention):
        se = torch.matmul(proj(module.wq_e), proj(module.wk_e).transpose(-2, -1))
        si = torch.matmul(proj(module.wq_i), proj(module.wk_i).transpose(-2, -1))
        se, si = se * scale + mask, si * scale + mask
        if sigmoid:
            b = module.attn_bias.view(1, -1, 1, 1)
            return {"exc": torch.sigmoid(se + b), "inhib": torch.sigmoid(si + b)}
        return {"exc": torch.softmax(se, -1), "inhib": torch.softmax(si, -1)}
    # MHAttention / SigmoidAttention
    s = torch.matmul(proj(module.wq), proj(module.wk).transpose(-2, -1)) * scale + mask
    if sigmoid:
        b = module.attn_bias.view(1, -1, 1, 1)
        return {"attn": torch.sigmoid(s + b)}
    return {"attn": torch.softmax(s, -1)}


def _effective_keys(w: torch.Tensor) -> torch.Tensor:
    """Perplexity of each row's normalised distribution → effective # keys."""
    p = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
    ent = -(p * p.clamp(min=1e-12).log()).sum(-1)
    return ent.exp()


def _overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Histogram intersection of two maps' normalised rows, per query (0..1)."""
    pa = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
    pb = b / b.sum(-1, keepdim=True).clamp(min=1e-9)
    return torch.minimum(pa, pb).sum(-1)


@torch.no_grad()
def probe_maps(model, name: str, x_ids: torch.Tensor) -> None:
    """Run one batch, recompute every attention layer's maps, report metrics."""
    captured: dict = {}

    def mk_hook(mod_name):
        def hook(_module, args):
            captured[mod_name] = (args[0].detach(), args[1].detach())

        return hook

    handles = [
        m.register_forward_pre_hook(mk_hook(n))
        for n, m in model.named_modules()
        if isinstance(m, (MHAttention, SignedAttention))
    ]
    model(x_ids)
    for h in handles:
        h.remove()

    mods = dict(model.named_modules())
    print(f"  [{name}]  attention map structure (queries in 2nd half of seq):")
    for mod_name, (x, freqs) in captured.items():
        T = x.shape[1]
        lo = T // 2  # only long-context queries — short prefixes are trivially peaked
        maps = attention_maps(mods[mod_name], x, freqs)
        if "attn" in maps:
            w = maps["attn"][..., lo:, :]
            eff = _effective_keys(w).mean().item()
            mass = w.sum(-1).mean().item()
            print(
                f"    {mod_name:24s} eff_keys {eff:6.2f}   "
                f"sigmoid_mass {mass:6.2f}"
            )
        else:
            e = maps["exc"][..., lo:, :]
            i = maps["inhib"][..., lo:, :]
            eff_e = _effective_keys(e).mean().item()
            eff_i = _effective_keys(i).mean().item()
            ov = _overlap(e, i).mean().item()
            extra = ""
            if maps["exc"].sum(-1).mean() > 1.5:  # sigmoid-style (unnormalised)
                extra = f"   mass exc {e.sum(-1).mean():.2f} / inhib {i.sum(-1).mean():.2f}"
            print(
                f"    {mod_name:24s} eff_keys exc {eff_e:6.2f} / inhib {eff_i:6.2f}"
                f"   exc-inhib overlap {ov:.3f}{extra}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ckpt-dir", default="variant_sweep_ckpts")
    p.add_argument("--sweep", default="sigmoid", help="checkpoint filename prefix")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    variants = ["mha", "sigmoid", "signed", "signed-sigmoid"]

    # one shared validation batch — identical input for every model
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    probe_path = f"{args.ckpt_dir}/{args.sweep}_mha.pt"
    seq_len = torch.load(probe_path, map_location="cpu", weights_only=False)[
        "cfg"
    ].max_seq_len
    ds = PackedLMDataset(
        load_text_ds(args.dataset, "", "validation"), tok, seq_len, 400_000
    )
    x_ids, _ = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=False)))
    x_ids = x_ids.to(device)
    print(f"[setup] device={device}  batch={tuple(x_ids.shape)}\n")

    bar = "=" * 74
    print(f"{bar}\nProbe 1 — learned inhibition gain γ\n{bar}")
    for v in ("signed", "signed-sigmoid"):
        model, _, name = load_model(f"{args.ckpt_dir}/{args.sweep}_{v}.pt", device)
        probe_inhibition_gain(model, name)

    print(f"\n{bar}\nProbe 2 — attention map structure\n{bar}")
    for v in variants:
        model, _, name = load_model(f"{args.ckpt_dir}/{args.sweep}_{v}.pt", device)
        probe_maps(model, name, x_ids)


if __name__ == "__main__":
    main()
