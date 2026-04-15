#!/usr/bin/env python3
"""
Conceptor Diagnostic Analysis for LIBERO Activations
=====================================================
Loads residual stream activations from local activation cache
(downloaded from brandonyang/pi05-libero-activations-v1-2000-15env)
and runs four diagnostic checks:

  (a) Singular value spectra of per-task conceptors
  (b) Conceptor similarity / overlap matrix across tasks
  (c) Boolean operations (AND, NOT) between conceptors
  (d) Linear probe validation on conceptor subspaces

Activation data lives at:
  $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000/

Each task has 15 episodes (episode_000_env_000 .. episode_014_env_000).
Activation tensors: suffix_residual.npz["all_suffix_residual"] shape (10, 4, 10, 1024)
  - 10 denoising steps
  - 4 captured layers [0, 5, 11, 17]
  - 10 action tokens (action_horizon=10)
  - 1024 hidden dim

Usage (from repo root):
    uv run experiments/pi05_libero/src/conceptor_diagnostic.py
    uv run experiments/pi05_libero/src/conceptor_diagnostic.py --layer 5
    uv run experiments/pi05_libero/src/conceptor_diagnostic.py --layer 17 --out_dir experiments/pi05_libero/diagnostic_results
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tyro
import dataclasses

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
ACTIVATIONS_ROOT = Path(OPENPI_DATA_HOME) / "activations" / "pi05_libero_2000_15env" / "openpi-libero-2000"

# Layer indices in the captured data: [0, 5, 11, 17]
LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}

DENOISE_STEPS = [0, 9]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]

TASK_COLORS = {}  # auto-assigned below

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ── Helpers ──────────────────────────────────────────────────────────────

def fast_svd(X, k=None):
    """SVD of mean-centred X. Returns eigenvalues of R = X^T X / N."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc / np.sqrt(max(1, N)), full_matrices=False)
    sigma = s ** 2
    if k is not None:
        return sigma[:k], Vt[:k]
    return sigma, Vt


def conceptor_eigenvalues(sigma, alpha):
    """gamma_j = sigma_j / (sigma_j + alpha^{-2})"""
    return sigma / (sigma + alpha ** -2)


def conceptor_quota(gamma):
    """q(C) = trace(C) = sum gamma_j"""
    return float(gamma.sum())


def conceptor_entropy(gamma):
    """H(C) = -sum [gamma log2 gamma + (1-gamma) log2(1-gamma)]"""
    g = np.clip(gamma, 1e-12, 1 - 1e-12)
    return float(-np.sum(g * np.log2(g) + (1 - g) * np.log2(1 - g)))


def conceptor_similarity(gamma_a, Vt_a, gamma_b, Vt_b, alpha):
    """Generalized similarity via inner product of conceptor matrices."""
    C_a = Vt_a.T @ np.diag(gamma_a) @ Vt_a
    C_b = Vt_b.T @ np.diag(gamma_b) @ Vt_b
    num = np.trace(C_a @ C_b)
    denom = np.sqrt(np.trace(C_a @ C_a) * np.trace(C_b @ C_b)) + 1e-12
    return float(num / denom)


def boolean_not_eigenvalues(gamma):
    return 1.0 - gamma


def boolean_and_quota(gamma_a, Vt_a, gamma_b, Vt_b):
    """Approximate AND quota using harmonic mean of eigenvalues."""
    C_a = Vt_a.T @ np.diag(gamma_a) @ Vt_a
    C_b = Vt_b.T @ np.diag(gamma_b) @ Vt_b
    d = C_a.shape[0]
    inner = C_a + C_b - C_a @ C_b + 1e-8 * np.eye(d)
    C_and = C_a @ np.linalg.inv(inner) @ C_b
    return float(np.trace(C_and))


# ── Data Loading ─────────────────────────────────────────────────────────

def discover_tasks():
    """Discover all tasks in the activation directory."""
    if not ACTIVATIONS_ROOT.exists():
        raise FileNotFoundError(
            f"Activation directory not found: {ACTIVATIONS_ROOT}\n"
            f"Download with: huggingface-cli download brandonyang/pi05-libero-activations-v1-2000-15env "
            f"--repo-type dataset --local-dir $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env"
        )
    tasks = sorted([d.name for d in ACTIVATIONS_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(tasks)} tasks:")
    for t in tasks:
        print(f"  {t}")
    return tasks


def load_episode_metadata(task, episode_name):
    """Load episode-level metadata.json."""
    meta_path = ACTIVATIONS_ROOT / task / episode_name / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def discover_episodes(task):
    """Find all episodes for a task and their success/failure status."""
    task_dir = ACTIVATIONS_ROOT / task
    episodes = sorted([d.name for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("episode")])
    success_eps, failure_eps = [], []
    for ep in episodes:
        meta = load_episode_metadata(task, ep)
        if meta is None:
            continue
        if meta.get("episode_success", False):
            success_eps.append(ep)
        else:
            failure_eps.append(ep)
    return success_eps, failure_eps


def load_activations_for_episode(task, episode_name, layer_idx):
    """Load residual stream activations for one episode.

    Returns:
        all_activations: dict mapping denoising_step -> np.array (n_inference, 10, 1024)
        metadata: episode metadata dict
    """
    meta = load_episode_metadata(task, episode_name)
    if meta is None:
        raise FileNotFoundError(f"No metadata for {task}/{episode_name}")

    n_inference = meta["total_inference_steps"]
    ep_dir = ACTIVATIONS_ROOT / task / episode_name

    # Discover step directories
    step_dirs = sorted([d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])

    all_activations = {t: [] for t in range(10)}

    for step_dir in step_dirs:
        res_path = step_dir / "suffix_residual.npz"
        if not res_path.exists():
            continue
        data = np.load(res_path)
        residual = data["all_suffix_residual"]  # (10, 4, 10, 1024)
        for t in range(10):
            act = residual[t, layer_idx, :, :]  # (10, 1024)
            all_activations[t].append(act)

    for t in range(10):
        if all_activations[t]:
            all_activations[t] = np.stack(all_activations[t])  # (n_steps, 10, 1024)
        else:
            all_activations[t] = np.zeros((0, 10, 1024))

    return all_activations, meta


def collect_task_activations(task, layer_idx, denoise_step=0):
    """Collect all activations for a task at a specific layer and denoise step.

    Returns: np.array (N, 1024) — flattened across episodes, inference steps, and action tokens.
    """
    success_eps, failure_eps = discover_episodes(task)
    all_eps = success_eps + failure_eps

    all_acts = []
    for ep in all_eps:
        try:
            acts, _ = load_activations_for_episode(task, ep, layer_idx)
            flat = acts[denoise_step].reshape(-1, 1024)
            all_acts.append(flat)
        except Exception as e:
            print(f"  Warning: skipping {task}/{ep}: {e}")
    if all_acts:
        return np.concatenate(all_acts, axis=0)
    return np.zeros((0, 1024))


# ── Diagnostic (a): Spectra ──────────────────────────────────────────────

def plot_spectra(tasks, layer_idx, denoise_step, out_dir):
    """Plot eigenvalue spectra of per-task conceptors."""
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4 * len(ALPHAS), 3.5), squeeze=False)

    for ai, alpha in enumerate(ALPHAS):
        ax = axes[0, ai]
        for task in tasks:
            X = collect_task_activations(task, layer_idx, denoise_step)
            if X.shape[0] == 0:
                continue
            sigma, _ = fast_svd(X)
            gamma = conceptor_eigenvalues(sigma, alpha)
            short_name = task.split("_", 2)[-1][:30]
            ax.plot(gamma[:100], label=short_name, color=TASK_COLORS.get(task, None), linewidth=1)
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("Index")
        if ai == 0:
            ax.set_ylabel("Eigenvalue")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Per-Task Conceptor Spectra (layer_idx={layer_idx}, denoise={denoise_step})", fontsize=11)
    fig.legend(
        [Line2D([0], [0], color=TASK_COLORS.get(t, "gray"), linewidth=1) for t in tasks],
        [t.split("_", 2)[-1][:30] for t in tasks],
        loc="lower center", ncol=min(5, len(tasks)), fontsize=6, bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_dir / f"spectra_denoise_{denoise_step}.png")
    plt.close()
    print(f"  Saved spectra_denoise_{denoise_step}.png")


# ── Diagnostic (b): Similarity Matrix ────────────────────────────────────

def plot_similarity(tasks, layer_idx, denoise_step, alpha, out_dir):
    """Plot pairwise conceptor similarity matrix."""
    n = len(tasks)
    sim_mat = np.zeros((n, n))
    task_data = {}

    for i, task in enumerate(tasks):
        X = collect_task_activations(task, layer_idx, denoise_step)
        if X.shape[0] == 0:
            continue
        sigma, Vt = fast_svd(X)
        gamma = conceptor_eigenvalues(sigma, alpha)
        task_data[i] = (gamma, Vt)

    for i in range(n):
        for j in range(n):
            if i in task_data and j in task_data:
                sim_mat[i, j] = conceptor_similarity(
                    task_data[i][0], task_data[i][1],
                    task_data[j][0], task_data[j][1], alpha
                )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlBu_r")
    short_names = [t.split("_", 2)[-1][:25] for t in tasks]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Conceptor Similarity (alpha={alpha}, denoise={denoise_step})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"similarity_denoise_{denoise_step}.png")
    plt.close()
    print(f"  Saved similarity_denoise_{denoise_step}.png")


# ── Diagnostic (c): Boolean Operations ───────────────────────────────────

def plot_boolean_ops(tasks, layer_idx, alpha, out_dir):
    """Plot quota values for AND and NOT operations between task pairs."""
    task_data = {}
    for task in tasks:
        X = collect_task_activations(task, layer_idx, denoise_step=0)
        if X.shape[0] == 0:
            continue
        sigma, Vt = fast_svd(X)
        gamma = conceptor_eigenvalues(sigma, alpha)
        task_data[task] = (gamma, Vt, conceptor_quota(gamma))

    if len(task_data) < 2:
        print("  Not enough tasks for boolean ops, skipping.")
        return

    task_list = list(task_data.keys())
    n = len(task_list)

    and_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            g_a, Vt_a, _ = task_data[task_list[i]]
            g_b, Vt_b, _ = task_data[task_list[j]]
            and_mat[i, j] = boolean_and_quota(g_a, Vt_a, g_b, Vt_b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AND matrix
    ax = axes[0]
    im = ax.imshow(and_mat, cmap="YlOrRd")
    short_names = [t.split("_", 2)[-1][:20] for t in task_list]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"AND Quota (alpha={alpha})")

    # NOT quotas
    ax = axes[1]
    quotas = [task_data[t][2] for t in task_list]
    not_quotas = [1024 - q for q in quotas]
    x_pos = range(len(task_list))
    ax.bar(x_pos, quotas, alpha=0.7, label="C quota")
    ax.bar(x_pos, not_quotas, bottom=quotas, alpha=0.7, label="NOT C quota")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Quota")
    ax.set_title(f"Quota vs NOT Quota (alpha={alpha})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"boolean_operations.png")
    plt.close()
    print(f"  Saved boolean_operations.png")


# ── Diagnostic (d): Linear Probe ─────────────────────────────────────────

def run_probes(tasks, layer_idx, out_dir):
    """Run linear probe classification: task ID, success/failure."""
    results = []

    # Task-ID probe (multi-class)
    print("  Running task-ID probe...")
    all_X, all_y = [], []
    for i, task in enumerate(tasks):
        X = collect_task_activations(task, layer_idx, denoise_step=0)
        if X.shape[0] == 0:
            continue
        # Subsample for speed
        if X.shape[0] > 2000:
            idx = np.random.choice(X.shape[0], 2000, replace=False)
            X = X[idx]
        all_X.append(X)
        all_y.extend([i] * X.shape[0])

    if all_X:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.array(all_y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X_all, y_all):
            clf = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="multinomial", C=1.0)
            clf.fit(X_all[train_idx], y_all[train_idx])
            accs.append(accuracy_score(y_all[test_idx], clf.predict(X_all[test_idx])))
        task_id_acc = np.mean(accs)
        results.append({"probe": "task_id", "accuracy": task_id_acc, "n_classes": len(tasks), "n_samples": len(y_all)})
        print(f"    Task-ID accuracy: {task_id_acc:.3f} ({len(tasks)}-way, {len(y_all)} samples)")

    # Success/failure probe (per-task binary)
    print("  Running success/failure probes...")
    for task in tasks:
        success_eps, failure_eps = discover_episodes(task)
        if not success_eps or not failure_eps:
            print(f"    {task}: no mixed outcomes, skipping")
            continue

        X_s, X_f = [], []
        for ep in success_eps:
            acts, _ = load_activations_for_episode(task, ep, layer_idx)
            X_s.append(acts[0].reshape(-1, 1024))
        for ep in failure_eps:
            acts, _ = load_activations_for_episode(task, ep, layer_idx)
            X_f.append(acts[0].reshape(-1, 1024))

        X_s = np.concatenate(X_s) if X_s else np.zeros((0, 1024))
        X_f = np.concatenate(X_f) if X_f else np.zeros((0, 1024))

        if X_s.shape[0] < 5 or X_f.shape[0] < 5:
            continue

        X = np.concatenate([X_s, X_f])
        y = np.array([1] * X_s.shape[0] + [0] * X_f.shape[0])

        skf = StratifiedKFold(n_splits=min(5, min(X_s.shape[0], X_f.shape[0])), shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=300, solver="lbfgs", C=1.0)
            clf.fit(X[train_idx], y[train_idx])
            accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))

        acc = np.mean(accs)
        short_name = task.split("_", 2)[-1][:40]
        results.append({
            "probe": "success_failure", "task": task, "accuracy": acc,
            "n_success": X_s.shape[0], "n_failure": X_f.shape[0],
        })
        print(f"    {short_name}: acc={acc:.3f} ({X_s.shape[0]}s/{X_f.shape[0]}f)")

    # Save results
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved probe_results.json")
    return results


# ── Quota-vs-Alpha Across Layers ─────────────────────────────────────────

def plot_quota_vs_alpha(tasks, out_dir):
    """Plot quota vs alpha for all layers and tasks."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    layer_names = {0: "Layer 0", 1: "Layer 5", 2: "Layer 11", 3: "Layer 17"}

    for li, (layer_model, layer_idx) in enumerate(LAYER_MAP.items()):
        ax = axes[li]
        for task in tasks:
            X = collect_task_activations(task, layer_idx, denoise_step=0)
            if X.shape[0] == 0:
                continue
            sigma, _ = fast_svd(X)
            quotas = [conceptor_quota(conceptor_eigenvalues(sigma, a)) for a in ALPHAS]
            short_name = task.split("_", 2)[-1][:25]
            ax.plot(ALPHAS, quotas, "o-", label=short_name, markersize=3, linewidth=1,
                    color=TASK_COLORS.get(task, None))
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_title(layer_names[layer_idx])
        if li == 0:
            ax.set_ylabel("Quota q(C)")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Quota vs Alpha Across Layers", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "quota_vs_alpha_all_layers.png")
    plt.close()
    print(f"  Saved quota_vs_alpha_all_layers.png")


# ── Main ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    layer: int = 11
    alpha: float = 1.0
    out_dir: str = "experiments/pi05_libero/diagnostic_results"


def main(args: Args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_idx = LAYER_MAP.get(args.layer)
    if layer_idx is None:
        raise ValueError(f"Invalid layer {args.layer}. Choose from {list(LAYER_MAP.keys())}")

    tasks = discover_tasks()

    # Assign colors
    cmap = plt.cm.get_cmap("tab10", len(tasks))
    for i, t in enumerate(tasks):
        TASK_COLORS[t] = cmap(i)

    print(f"\nRunning diagnostics: layer={args.layer} (idx={layer_idx}), alpha={args.alpha}")
    print(f"Output: {out_dir}\n")

    # (a) Spectra
    print("[a] Eigenvalue spectra...")
    for ds in DENOISE_STEPS:
        plot_spectra(tasks, layer_idx, ds, out_dir)

    # (b) Similarity
    print("[b] Similarity matrix...")
    for ds in DENOISE_STEPS:
        plot_similarity(tasks, layer_idx, ds, args.alpha, out_dir)

    # (c) Boolean operations
    print("[c] Boolean operations...")
    plot_boolean_ops(tasks, layer_idx, args.alpha, out_dir)

    # (d) Probes
    print("[d] Linear probes...")
    run_probes(tasks, layer_idx, out_dir)

    # Quota vs alpha across layers
    print("[e] Quota vs alpha across layers...")
    plot_quota_vs_alpha(tasks, out_dir)

    # Summary: success/failure breakdown per task
    print("\n=== Task Outcome Summary ===")
    for task in tasks:
        s, f = discover_episodes(task)
        short = task.split("_", 2)[-1][:50]
        print(f"  {short}: {len(s)} success, {len(f)} failure")

    print(f"\nAll diagnostics saved to {out_dir}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
