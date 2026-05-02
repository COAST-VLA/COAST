"""
Shared utilities for mechanistic interpretability analyses.
Data loading, conceptor math, plotting style, and common constants.
"""

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging
import csv

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ID = "brandonyang/ml45-activations-15"
CHECKPOINT = "5000"
HF_CACHE = "/nlp/data/huggingface_cache"
NUM_ENVS = 15
NUM_DENOISE_STEPS = 10
HIDDEN_DIM = 1024
NUM_ACTION_TOKENS = 32

LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}
LAYER_NAMES = {0: "L0", 1: "L5", 2: "L11", 3: "L17"}

STEERING_RESULTS_DIR = Path("/nlpgpu/data/miaom/openpi-metaworld/experiments/steering_results")
OUTPUT_DIR = Path("/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis/results")

# 26 mixed-outcome tasks
MIXED_TASKS = [
    "assembly-v3", "basketball-v3", "coffee-pull-v3", "coffee-push-v3",
    "disassemble-v3", "door-open-v3", "faucet-close-v3", "hammer-v3",
    "handle-pull-side-v3", "handle-pull-v3", "lever-pull-v3",
    "peg-insert-side-v3", "pick-out-of-hole-v3", "pick-place-v3",
    "pick-place-wall-v3", "plate-slide-back-side-v3", "plate-slide-back-v3",
    "push-back-v3", "push-v3", "reach-v3", "shelf-place-v3", "soccer-v3",
    "stick-pull-v3", "stick-push-v3", "sweep-into-v3", "sweep-v3",
]

# ── NeurIPS Plotting Style ───────────────────────────────────────────────────

COLORS = {
    'teal':  '#4C9F8B',
    'coral': '#E07A5F',
    'gold':  '#D4A843',
    'slate': '#5B7FA5',
    'rose':  '#C97B84',
    'dark':  '#2D3142',
}

# Ordered palette for cycling through tasks
TASK_PALETTE = [
    '#4C9F8B', '#E07A5F', '#D4A843', '#5B7FA5', '#C97B84',
    '#6A9F7B', '#D06A4F', '#C49833', '#4B6F95', '#B96B74',
    '#7AB09B', '#F08A6F', '#E4B853', '#6B8FB5', '#D98B94',
    '#3A8F6B', '#C05A3F', '#B48823', '#3B5F85', '#A95B64',
    '#8AC0AB', '#FF9A7F', '#F4C863', '#7B9FC5', '#E99BA4',
    '#2A7F5B',
]


def apply_neurips_style():
    """Apply global NeurIPS-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
    })


# ── Data Loading ─────────────────────────────────────────────────────────────

def find_steerable_tasks(tasks=None):
    """Find tasks with success/failure splits from HuggingFace metadata.

    Returns dict: task -> {"success": [env_names], "failure": [env_names], "has_failures": bool}
    """
    if tasks is None:
        tasks = MIXED_TASKS

    steerable = {}
    for task in tqdm(tasks, desc="Finding steerable tasks"):
        success_envs, failure_envs = [], []
        for env_idx in range(NUM_ENVS):
            env_name = f"episode_000_env_{env_idx:03d}"
            try:
                meta_path = hf_hub_download(
                    REPO_ID, f"{CHECKPOINT}/{task}/{env_name}/metadata.json",
                    repo_type="dataset", cache_dir=HF_CACHE,
                )
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["episode_success"]:
                    success_envs.append(env_name)
                else:
                    failure_envs.append(env_name)
            except Exception:
                continue
        if success_envs and failure_envs:
            steerable[task] = {"success": success_envs, "failure": failure_envs, "has_failures": True}
        elif success_envs:
            steerable[task] = {"success": success_envs, "failure": [], "has_failures": False}
    return steerable


def load_activations_for_episode(task, env_name, layer_idx=2):
    """Load residual stream activations for one episode from HuggingFace.

    Returns:
        all_activations: dict {denoise_step: np.array (n_inference, 32, 1024)}
        metadata: episode metadata dict
    """
    meta_path = hf_hub_download(
        REPO_ID, f"{CHECKPOINT}/{task}/{env_name}/metadata.json",
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    with open(meta_path) as f:
        meta = json.load(f)

    n_inference = meta["total_inference_steps"]
    all_activations = {t: [] for t in range(NUM_DENOISE_STEPS)}

    for step in range(n_inference):
        step_name = f"step_{step * 10:04d}"
        res_path = hf_hub_download(
            REPO_ID, f"{CHECKPOINT}/{task}/{env_name}/{step_name}/suffix_residual.npz",
            repo_type="dataset", cache_dir=HF_CACHE,
        )
        data = np.load(res_path)
        residual = data["all_suffix_residual"]  # (10, 4, 32, 1024)
        for t in range(NUM_DENOISE_STEPS):
            all_activations[t].append(residual[t, layer_idx, :, :])  # (32, 1024)

    for t in range(NUM_DENOISE_STEPS):
        all_activations[t] = np.stack(all_activations[t])  # (n_inference, 32, 1024)

    return all_activations, meta


def collect_outcome_activations(task, env_splits, layer_idx=2, mean_pool=True):
    """Collect activations grouped by success/failure.

    Args:
        mean_pool: if True, mean-pool over 32 action tokens -> (N, 1024).
                   if False, flatten -> (N*32, 1024).

    Returns:
        success_acts: dict {denoise_step: np.array (N, 1024)}
        failure_acts: dict {denoise_step: np.array (N, 1024)}
    """
    success_acts = {t: [] for t in range(NUM_DENOISE_STEPS)}
    failure_acts = {t: [] for t in range(NUM_DENOISE_STEPS)}

    for outcome, env_list in [("success", env_splits["success"]),
                               ("failure", env_splits["failure"])]:
        target = success_acts if outcome == "success" else failure_acts
        for env_name in tqdm(env_list, desc=f"  {outcome}", leave=False):
            acts, meta = load_activations_for_episode(task, env_name, layer_idx)
            for t in range(NUM_DENOISE_STEPS):
                if mean_pool:
                    pooled = acts[t].mean(axis=1)  # (n_inference, 1024)
                else:
                    pooled = acts[t].reshape(-1, HIDDEN_DIM)  # (n_inference*32, 1024)
                target[t].append(pooled)

    for t in range(NUM_DENOISE_STEPS):
        if success_acts[t]:
            success_acts[t] = np.concatenate(success_acts[t], axis=0)
        else:
            success_acts[t] = np.zeros((0, HIDDEN_DIM))
        if failure_acts[t]:
            failure_acts[t] = np.concatenate(failure_acts[t], axis=0)
        else:
            failure_acts[t] = np.zeros((0, HIDDEN_DIM))

    return success_acts, failure_acts


# ── Conceptor Math ───────────────────────────────────────────────────────────

def compute_conceptor(X, alpha=1.0):
    """Compute conceptor C = R (R + alpha^{-2} I)^{-1}.

    Returns: (C, eigenvalues_descending)
    """
    d = X.shape[1]
    R = (X.T @ X) / X.shape[0]
    reg = (alpha ** -2) * np.eye(d)
    C = R @ np.linalg.inv(R + reg)
    eigenvalues = np.linalg.eigvalsh(C)[::-1]
    return C, eigenvalues


def boolean_and(C_A, C_B):
    """Soft intersection: C_A AND C_B."""
    d = C_A.shape[0]
    inner = C_A + C_B - C_A @ C_B + 1e-8 * np.eye(d)
    return C_A @ np.linalg.inv(inner) @ C_B


def boolean_not(C):
    """Soft complement: NOT C = I - C."""
    return np.eye(C.shape[0]) - C


def contrastive_conceptor(C_positive, C_negative):
    """C_positive AND (NOT C_negative)."""
    return boolean_and(C_positive, boolean_not(C_negative))


def conceptor_quota(C):
    """Quota = trace(C) = sum of eigenvalues."""
    return np.trace(C)


def conceptor_overlap(C_A, C_B):
    """Overlap = tr(C_A @ C_B) / tr(C_A)."""
    tr_A = np.trace(C_A)
    if tr_A < 1e-10:
        return 0.0
    return np.trace(C_A @ C_B) / tr_A


def effective_rank(eigenvalues):
    """Effective rank via entropy of normalized eigenvalues."""
    eigs = np.clip(eigenvalues, 1e-12, None)
    p = eigs / eigs.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


# ── Steering Results Loading ─────────────────────────────────────────────────

def load_steering_csv(task):
    """Load steering results CSV for a task. Returns list of dicts."""
    csv_path = STEERING_RESULTS_DIR / task / f"results_{task}.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_diagnostics(task, alpha=0.5):
    """Load conceptor diagnostics JSON for a task."""
    diag_path = STEERING_RESULTS_DIR / task / f"diagnostics_a{alpha}.json"
    if not diag_path.exists():
        return None
    with open(diag_path) as f:
        return json.load(f)


def get_baseline_sr(results_rows):
    """Extract baseline success rate from results CSV rows."""
    for row in results_rows:
        if row["condition"] == "baseline":
            return float(row["success_rate"])
    return None


def get_best_conceptor_sr(results_rows, strategy="strategy3"):
    """Get best conceptor success rate across alpha/beta for a given strategy."""
    best = -1.0
    for row in results_rows:
        cond = row["condition"]
        if cond.startswith(strategy + "_"):
            sr = float(row["success_rate"])
            if sr > best:
                best = sr
    return best if best >= 0 else None


def get_best_linear_sr(results_rows):
    """Get best linear steering success rate."""
    best = -1.0
    for row in results_rows:
        if row["condition"].startswith("linear_"):
            sr = float(row["success_rate"])
            if sr > best:
                best = sr
    return best if best >= 0 else None


def ensure_output_dir():
    """Create output directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
