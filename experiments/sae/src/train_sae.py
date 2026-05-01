#!/usr/bin/env python3
"""Train a per-task TopK SAE on captured residual-stream activations.

Activation schemas (captured at collection time, see experiments/{exp}/src/fit_linear_vectors.py):

  pi05_libero / pi05_robocasa  (PyTorch flow-matching policies):
    {task}/episode_*/step_*/suffix_residual.npz
        key  = "all_suffix_residual"
        shape = (num_denoising_steps=10, num_captured_layers=4, seq_len, hidden=1024)
        layer_map = {0: 0, 5: 1, 11: 2, 17: 3}

  pi0_fast_libero / pi0_fast_metaworld  (JAX autoregressive):
    {task}/episode_*/step_*/hidden_states.npz
        key  = "token_pre_logits"
        shape = (n_tokens, hidden=2048)
        no layer / denoising-step axis — single intervention point.

For pi05* we train ONE SAE per (task, layer) — layers come from --layers (default
{0,5,11,17}). For pi0_fast* we train ONE SAE per task — pass --layers '' to skip
the layer axis.

Recipes are per-task (matching the conceptor pipeline). Output:
    {output_dir}/{task}__L{L}.pt        (pi05*)
    {output_dir}/{task}.pt              (pi0_fast*)
each storing a dict with keys: state_dict, config, holdout_var_explained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Make sibling module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_module import TopKSAE  # noqa: E402

OPENPI_DATA_HOME = Path(os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache" / "openpi")))

# ── Activation loaders (one per schema) ──────────────────────────────────────


def discover_tasks(root: Path, denylist=frozenset({".cache"})) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in denylist)


def load_episode_metadata(task_dir: Path) -> dict:
    info = {}
    for ep_dir in sorted(task_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        info[ep_dir.name] = {"path": ep_dir, "success": bool(meta.get("episode_success", False))}
    return info


def load_pi05_task_vectors(task_dir: Path, layer_idx: int) -> np.ndarray:
    """Pi0.5 PyTorch suffix-residual schema. Returns (n_samples, 1024).

    Pools across all (denoising_step, action_token) for a fixed layer — the SAE
    sees a single mixed distribution per task. Per-step variants are computed
    later in fit_sae_vectors.py without retraining.
    """
    info = load_episode_metadata(task_dir)
    chunks: list[np.ndarray] = []
    for _, ep in info.items():
        for step_dir in sorted(ep["path"].glob("step_*")):
            npz = step_dir / "suffix_residual.npz"
            if not npz.is_file():
                continue
            try:
                with np.load(npz) as f:
                    arr = f["all_suffix_residual"]  # (10, 4, seq, 1024)
                    sub = arr[:, layer_idx]  # (10, seq, 1024)
                    chunks.append(sub.reshape(-1, sub.shape[-1]).astype(np.float32))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {npz}: {e}", file=sys.stderr)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 1024), dtype=np.float32)


def load_pi0fast_task_vectors(task_dir: Path) -> np.ndarray:
    """Pi0-fast token-pre-logits schema. Returns (n_tokens, 2048)."""
    chunks: list[np.ndarray] = []
    info = load_episode_metadata(task_dir)
    for _, ep in info.items():
        for step_dir in sorted(ep["path"].glob("step_*")):
            hp = step_dir / "hidden_states.npz"
            if not hp.is_file():
                continue
            try:
                with np.load(hp) as data:
                    chunks.append(data["token_pre_logits"].astype(np.float32))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {hp}: {e}", file=sys.stderr)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2048), dtype=np.float32)


# ── Training loop ────────────────────────────────────────────────────────────


def train_one_sae(
    X: np.ndarray,
    d_sae: int,
    k: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int = 0,
    holdout_frac: float = 0.05,
) -> tuple[TopKSAE, dict]:
    """Train a TopKSAE on X (n, d_model). Returns (model, diagnostics)."""
    rng = np.random.default_rng(seed)
    n, d_model = X.shape
    perm = rng.permutation(n)
    n_holdout = max(1, int(round(n * holdout_frac)))
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    X_train = torch.from_numpy(X[train_idx]).to(device)
    X_hold = torch.from_numpy(X[holdout_idx]).to(device)

    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
        h = X_train[idx]
        h_hat, _ = sae(h)
        loss = ((h - h_hat) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        losses.append(loss.item())
        if (step + 1) % max(1, n_steps // 10) == 0:
            print(f"    step {step+1:>5d}/{n_steps}  recon_mse={np.mean(losses[-200:]):.5f}")

    # Hold-out variance-explained (closer to 1.0 = better recon).
    sae.eval()
    with torch.no_grad():
        h_hat, _ = sae(X_hold)
        ss_res = ((X_hold - h_hat) ** 2).sum().item()
        ss_tot = ((X_hold - X_hold.mean(0, keepdim=True)) ** 2).sum().item()
        var_exp = 1.0 - ss_res / max(ss_tot, 1e-12)
    return sae, {
        "final_train_loss": float(np.mean(losses[-200:])),
        "holdout_var_explained": float(var_exp),
        "n_train": int(X_train.shape[0]),
        "n_holdout": int(X_hold.shape[0]),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schema", required=True, choices=["pi05", "pi0fast"],
                   help="pi05 = (10,4,seq,1024) suffix_residual; pi0fast = (n,2048) token_pre_logits")
    p.add_argument("--activations-dir", type=Path, required=True,
                   help="directory containing per-task subdirs")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to save {task}__L{L}.pt / {task}.pt")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="pi05 only: layer indices to fit (default 0 5 11 17). For pi0fast pass nothing.")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="optional task subset. Default: all discovered.")
    p.add_argument("--d-sae-mult", type=int, default=4,
                   help="d_sae = mult × hidden. Default 4 (so 4096 for pi05, 8192 for pi0fast).")
    p.add_argument("--k", type=int, default=64, help="TopK sparsity (fixed across the paper).")
    p.add_argument("--n-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--min-samples", type=int, default=4096,
                   help="skip tasks with fewer than this many activation vectors")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.activations_dir.is_dir():
        sys.exit(f"activations dir not found: {args.activations_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers
    if args.schema == "pi05":
        if layers is None:
            layers = [0, 5, 11, 17]
        layer_map = {0: 0, 5: 1, 11: 2, 17: 3}  # only valid layer names for pi05
        for L in layers:
            if L not in layer_map:
                sys.exit(f"layer {L} not in {list(layer_map)}")
    else:  # pi0fast
        if layers:
            print("WARNING: --layers ignored for pi0fast schema (single intervention point)")
        layers = [None]

    tasks = args.tasks or discover_tasks(args.activations_dir)
    print(f"schema={args.schema}  tasks={len(tasks)}  layers={layers}")
    print(f"output_dir={args.output_dir}")

    summary_path = args.output_dir / "training_summary.json"
    summary = {}
    if summary_path.is_file():
        with open(summary_path) as f:
            summary = json.load(f)

    for task in tasks:
        task_dir = args.activations_dir / task
        if not task_dir.is_dir():
            print(f"[{task[:60]}] task dir missing — skip")
            continue
        for L in layers:
            tag = f"{task}__L{L}" if args.schema == "pi05" else task
            out_path = args.output_dir / f"{tag}.pt"
            if out_path.is_file() and not args.overwrite:
                print(f"[{tag[:70]}] exists — skip (use --overwrite to refit)")
                continue
            print(f"\n[{tag[:80]}] loading activations...")
            t0 = time.time()
            if args.schema == "pi05":
                X = load_pi05_task_vectors(task_dir, layer_map[L])
                d_model = 1024
            else:
                X = load_pi0fast_task_vectors(task_dir)
                d_model = 2048
            print(f"  loaded n={X.shape[0]} d={X.shape[1]} in {time.time()-t0:.1f}s")
            if X.shape[0] < args.min_samples:
                print(f"  too few samples ({X.shape[0]} < {args.min_samples}) — skip")
                continue
            if X.shape[1] != d_model:
                print(f"  unexpected d_model={X.shape[1]} (want {d_model}) — skip")
                continue

            d_sae = args.d_sae_mult * d_model
            print(f"  training TopK SAE  d_sae={d_sae}  k={args.k}  n_steps={args.n_steps}")
            sae, diag = train_one_sae(
                X=X, d_sae=d_sae, k=args.k,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, device=args.device, seed=args.seed,
            )
            print(f"  holdout_var_explained = {diag['holdout_var_explained']:.4f}")

            payload = {
                "state_dict": {k_: v.detach().cpu() for k_, v in sae.state_dict().items()},
                "config": {
                    "schema": args.schema, "task": task, "layer": L,
                    "d_model": d_model, "d_sae": d_sae, "k": args.k,
                    "n_steps": args.n_steps, "batch_size": args.batch_size, "lr": args.lr,
                    "seed": args.seed,
                },
                "diag": diag,
            }
            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(out_path)
            summary[tag] = diag
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
