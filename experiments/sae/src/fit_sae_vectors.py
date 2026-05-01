#!/usr/bin/env python3
"""Build per-task SAE steering vectors using the Khan-style filter recipe.

Pipeline:
  1. Load the per-task SAE trained by train_sae.py.
  2. Encode every captured activation h to its sparse code f = TopK(ReLU(...))
     and split by episode_success.
  3. Per-feature stats:
        mu_pos[k] = mean of f_k over positive (success) samples
        mu_neg[k] = mean of f_k over negative (failure) samples
        fire_rate[k] = fraction of (pos∪neg) samples with f_k > 0
  4. Two filters:
        Filter A (rare):              keep if fire_rate >= --rare-thresh   (0.005)
        Filter B (bilaterally active): keep if min(mu_pos,mu_neg)
                                                  / max(mu_pos,mu_neg) <= --bilateral-thresh  (0.5)
     (mu's are non-negative because the SAE encoder is ReLU→TopK.)
  5. Steering vector:
        v_latent = (mu_pos - mu_neg) on surviving features, else 0
        v_sae    = W_dec^T @ v_latent   (W_dec stored as (d_sae, d_model))
        v_sae   /= ||v_sae||            (unit-norm — matches fit_linear_vectors.py)
  6. Write v_sae to a small NPZ alongside the linear-vectors NPZ, with key naming
     that matches the existing ActAdd loaders so the steering driver can pull
     either vector with the same lookup pattern.

NPZ key naming:
  pi05_libero / pi05_robocasa:  {task}__L{layer}__sae__V_contrastive   (1024,)
  pi0_fast_libero / metaworld:  {task}__sae__V_contrastive             (2048,)

Per-denoising-step variants (per_step_0, per_step_9) are NOT computed — for an
SAE basis they're just the same W_dec with means estimated from 1/10 the
samples, which adds variance without probing a different mechanism. Mirrors
what `linear_final` does: one global vector per task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_module import TopKSAE  # noqa: E402

OPENPI_DATA_HOME = Path(os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache" / "openpi")))


# ── Activation loaders (matched to the trainer) ──────────────────────────────


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


def _stack(chunks_pos, chunks_neg, d):
    Xp = np.concatenate(chunks_pos, axis=0) if chunks_pos else np.empty((0, d), dtype=np.float32)
    Xn = np.concatenate(chunks_neg, axis=0) if chunks_neg else np.empty((0, d), dtype=np.float32)
    return Xp, Xn


def load_pi05_pos_neg(task_dir: Path, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_pos, X_neg) of shape (n, 1024). Pools over all 10 denoising steps."""
    chunks_pos: list[np.ndarray] = []
    chunks_neg: list[np.ndarray] = []
    for ep in load_episode_metadata(task_dir).values():
        target = chunks_pos if ep["success"] else chunks_neg
        for step_dir in sorted(ep["path"].glob("step_*")):
            npz = step_dir / "suffix_residual.npz"
            if not npz.is_file():
                continue
            try:
                with np.load(npz) as f:
                    arr = f["all_suffix_residual"]  # (10, 4, seq, 1024)
                    sub = arr[:, layer_idx]  # (10, seq, 1024)
                    target.append(sub.reshape(-1, sub.shape[-1]).astype(np.float32))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {npz}: {e}", file=sys.stderr)
    return _stack(chunks_pos, chunks_neg, 1024)


def load_pi0fast_pos_neg(task_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    chunks_pos: list[np.ndarray] = []
    chunks_neg: list[np.ndarray] = []
    for ep in load_episode_metadata(task_dir).values():
        target = chunks_pos if ep["success"] else chunks_neg
        per_step: list[np.ndarray] = []
        for step_dir in sorted(ep["path"].glob("step_*")):
            hp = step_dir / "hidden_states.npz"
            if not hp.is_file():
                continue
            try:
                with np.load(hp) as data:
                    per_step.append(data["token_pre_logits"].astype(np.float32))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {hp}: {e}", file=sys.stderr)
        if per_step:
            target.append(np.concatenate(per_step, axis=0))
    return _stack(chunks_pos, chunks_neg, 2048)


# ── Encoding + filtering ─────────────────────────────────────────────────────


def encode_in_batches(sae: TopKSAE, X: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    """Encode (n, d_model) → (n, d_sae). Stays on CPU at the end."""
    if X.shape[0] == 0:
        return np.zeros((0, sae.d_sae), dtype=np.float32)
    out = np.empty((X.shape[0], sae.d_sae), dtype=np.float32)
    sae.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            chunk = torch.from_numpy(X[i:i + batch]).to(device)
            f = sae.encode(chunk)
            out[i:i + batch] = f.cpu().numpy()
    return out


def filter_and_build_v(
    f_pos: np.ndarray, f_neg: np.ndarray, W_dec: np.ndarray,
    rare_thresh: float, bilateral_thresh: float,
    fire_rate_global: np.ndarray | None = None,
):
    """Returns (v_sae, diagnostics). v_sae is unit-normalized.

    fire_rate_global: optional pre-computed global fire-rate to use for Filter A
    (used so per-step variants share the same rare-feature mask).
    """
    mu_pos = f_pos.mean(axis=0) if f_pos.shape[0] else np.zeros(f_neg.shape[1] if f_neg.size else 0, dtype=np.float32)
    mu_neg = f_neg.mean(axis=0) if f_neg.shape[0] else np.zeros(mu_pos.shape, dtype=np.float32)
    if fire_rate_global is not None:
        fire_rate = fire_rate_global
    else:
        n = f_pos.shape[0] + f_neg.shape[0]
        fire_pos = (f_pos > 0).sum(axis=0) if f_pos.shape[0] else 0
        fire_neg = (f_neg > 0).sum(axis=0) if f_neg.shape[0] else 0
        fire_rate = (fire_pos + fire_neg) / max(n, 1)

    keep = fire_rate >= rare_thresh
    n_after_A = int(keep.sum())

    # Filter B: bilaterally active. Uses non-negative means (ReLU/TopK).
    eps = 1e-8
    bilateral = np.minimum(mu_pos, mu_neg) / (np.maximum(mu_pos, mu_neg) + eps)
    keep &= bilateral <= bilateral_thresh
    n_after_B = int(keep.sum())

    v_latent = np.where(keep, mu_pos - mu_neg, 0.0).astype(np.float32)
    v_sae = (v_latent @ W_dec).astype(np.float32)  # (d_sae,) @ (d_sae, d_model) -> (d_model,)
    raw_norm = float(np.linalg.norm(v_sae))
    if raw_norm > 1e-12:
        v_sae = v_sae / raw_norm

    # Top-5 contributing features for diagnostics.
    contrib = (v_latent ** 2)  # squared contribution proxy
    top_idx = np.argsort(-contrib)[:5].tolist()
    diag = {
        "n_pos": int(f_pos.shape[0]),
        "n_neg": int(f_neg.shape[0]),
        "n_after_A": n_after_A,
        "n_after_B": n_after_B,
        "raw_v_norm": raw_norm,
        "top_features": [int(k) for k in top_idx],
        "top_features_delta": [float(v_latent[k]) for k in top_idx],
    }
    return v_sae, diag


# ── Driver ────────────────────────────────────────────────────────────────────


def load_sae(pt_path: Path, device: str) -> TopKSAE:
    payload = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    sae = TopKSAE(d_model=cfg["d_model"], d_sae=cfg["d_sae"], k=cfg["k"]).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def discover_tasks(root: Path, denylist=frozenset({".cache"})) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in denylist)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schema", required=True, choices=["pi05", "pi0fast"])
    p.add_argument("--activations-dir", type=Path, required=True)
    p.add_argument("--sae-dir", type=Path, required=True,
                   help="dir containing per-task SAE checkpoints from train_sae.py")
    p.add_argument("--output-npz", type=Path, required=True,
                   help="output NPZ (e.g. $OPENPI_DATA_HOME/libero_sae_vectors.npz)")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="pi05 only: layers to fit. Default 0 5 11 17.")
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    p.add_argument("--rare-thresh", type=float, default=0.005)
    p.add_argument("--bilateral-thresh", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--var-explained-floor", type=float, default=0.80,
                   help="skip tasks where the SAE's holdout var-explained is below this")
    args = p.parse_args()

    if args.schema == "pi05":
        layers = args.layers if args.layers is not None else [0, 5, 11, 17]
        layer_map = {0: 0, 5: 1, 11: 2, 17: 3}
    else:
        layers = [None]

    tasks = args.tasks or discover_tasks(args.activations_dir)
    print(f"schema={args.schema}  tasks={len(tasks)}  layers={layers}")
    print(f"output_npz={args.output_npz}")

    # Append-mode NPZ (matches fit_linear_vectors.py pattern).
    existing: dict[str, np.ndarray] = {}
    if args.output_npz.is_file():
        with np.load(args.output_npz) as z:
            existing = {k: z[k] for k in z.keys()}
        print(f"loaded {len(existing)} existing keys")

    diag_path = args.output_npz.with_suffix(".diagnostics.json")
    diagnostics = json.loads(diag_path.read_text()) if diag_path.is_file() else {}

    # Load training-summary for var-explained gate.
    train_summary_path = args.sae_dir / "training_summary.json"
    train_summary = json.loads(train_summary_path.read_text()) if train_summary_path.is_file() else {}

    n_added = 0
    for task in tasks:
        task_dir = args.activations_dir / task
        if not task_dir.is_dir():
            continue
        for L in layers:
            tag = f"{task}__L{L}" if args.schema == "pi05" else task
            sae_path = args.sae_dir / f"{tag}.pt"
            if not sae_path.is_file():
                print(f"[{tag[:70]}] no SAE checkpoint — skip")
                continue
            ve = train_summary.get(tag, {}).get("holdout_var_explained")
            if ve is not None and ve < args.var_explained_floor:
                print(f"[{tag[:70]}] var_explained={ve:.3f} < {args.var_explained_floor} — skip")
                diagnostics[tag] = {"skipped_var_explained": ve}
                continue
            print(f"\n[{tag[:80]}] loading activations...")
            sae = load_sae(sae_path, device=args.device)

            if args.schema == "pi05":
                X_pos, X_neg = load_pi05_pos_neg(task_dir, layer_map[L])
            else:
                X_pos, X_neg = load_pi0fast_pos_neg(task_dir)
            print(f"  n_pos={X_pos.shape[0]} n_neg={X_neg.shape[0]}")
            if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
                print("  insufficient data — skip")
                continue

            f_pos = encode_in_batches(sae, X_pos, args.device)
            f_neg = encode_in_batches(sae, X_neg, args.device)
            W_dec = sae.W_dec.detach().cpu().numpy()  # (d_sae, d_model)

            v, diag = filter_and_build_v(
                f_pos, f_neg, W_dec, args.rare_thresh, args.bilateral_thresh,
            )
            key = f"{task}__L{L}__sae__V_contrastive" if args.schema == "pi05" else f"{task}__sae__V_contrastive"
            existing[key] = v.astype(np.float32)
            diagnostics[tag] = diag
            n_added += 1
            print(f"  kept {diag['n_after_B']}/{f_pos.shape[1]} features  "
                  f"||v||_pre={diag['raw_v_norm']:.3f}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_npz.parent / f"{args.output_npz.stem}.tmp{args.output_npz.suffix}"
    np.savez_compressed(tmp, **existing)
    tmp.replace(args.output_npz)
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    print(f"\nwrote {len(existing)} keys ({n_added} new) → {args.output_npz}  "
          f"({args.output_npz.stat().st_size/1e6:.2f} MB)")
    print(f"diagnostics → {diag_path}")


if __name__ == "__main__":
    main()
