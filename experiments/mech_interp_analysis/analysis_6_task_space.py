#!/usr/bin/env python3
"""
Analysis 6: Task Space Structure — Conceptor Similarity & UMAP Embedding.

Panel A: Clustered conceptor similarity matrix with dendrogram (seaborn clustermap).
Panel B: 2D UMAP embedding of task conceptors colored by semantic category.

Computes contrastive conceptors C_steer for all 26 tasks (pooling across denoising
steps at layer 11), then visualizes inter-task structure.

Saves:
  - analysis_6_clustermap.pdf / .png         (Panel A standalone)
  - analysis_6_umap.pdf / .png               (Panel B standalone)
  - analysis_6_task_space.pdf / .png          (Combined 2-panel figure)
  - analysis_6_data.json                      (similarity matrix, linkage, etc.)
"""

import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
sys.path.insert(0, "/nlpgpu/data/miaom/activation_inform/.packages")

from shared_utils import (
    apply_neurips_style, COLORS, ensure_output_dir,
    MIXED_TASKS, HIDDEN_DIM,
    find_steerable_tasks, collect_outcome_activations,
    compute_conceptor, contrastive_conceptor,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ── Semantic category mapping ─────────────────────────────────────────────────

CATEGORY_MAP = {
    # Reaching / pressing
    "reach-v3": "reach/press",
    "stick-push-v3": "reach/press",
    "push-back-v3": "reach/press",

    # Pushing / sweeping
    "sweep-v3": "push/sweep",
    "sweep-into-v3": "push/sweep",
    "push-v3": "push/sweep",
    "coffee-push-v3": "push/sweep",
    "plate-slide-back-side-v3": "push/sweep",
    "plate-slide-back-v3": "push/sweep",
    "stick-pull-v3": "push/sweep",

    # Pick and place
    "pick-place-v3": "pick-and-place",
    "pick-place-wall-v3": "pick-and-place",
    "pick-out-of-hole-v3": "pick-and-place",
    "shelf-place-v3": "pick-and-place",
    "basketball-v3": "pick-and-place",

    # Articulated objects
    "door-open-v3": "articulated",
    "faucet-close-v3": "articulated",
    "lever-pull-v3": "articulated",

    # Handle / peg manipulation
    "handle-pull-v3": "handle/peg",
    "handle-pull-side-v3": "handle/peg",
    "peg-insert-side-v3": "handle/peg",

    # Assembly / insertion
    "assembly-v3": "assembly",
    "disassemble-v3": "assembly",

    # Dexterous / in-hand
    "coffee-pull-v3": "dexterous",
    "soccer-v3": "dexterous",
    "hammer-v3": "dexterous",
}

CATEGORY_COLORS = {
    "reach/press": "#2a9d8f",     # teal
    "push/sweep": "#e9c46a",      # gold
    "pick-and-place": "#264653",   # dark slate
    "articulated": "#e76f51",      # burnt orange
    "handle/peg": "#606c38",       # olive
    "assembly": "#9b2226",         # dark red
    "dexterous": "#457b9d",        # steel blue
}

ALPHA = 0.5  # Aperture for conceptor computation


def compute_all_conceptors(steerable_tasks, layer_idx=2):
    """Compute contrastive conceptor C_steer for each task (pooled across steps)."""
    conceptors = {}
    for i, (task, splits) in enumerate(steerable_tasks.items()):
        logger.info(f"[{i+1}/{len(steerable_tasks)}] {task}: loading activations...")
        success_acts, failure_acts = collect_outcome_activations(task, splits, layer_idx,
                                                                  mean_pool=False)
        # Pool across all denoising steps
        s_all = np.concatenate([success_acts[t] for t in range(10)], axis=0)
        f_all = np.concatenate([failure_acts[t] for t in range(10)], axis=0)

        C_pos, _ = compute_conceptor(s_all, alpha=ALPHA)
        C_neg, _ = compute_conceptor(f_all, alpha=ALPHA)
        C_steer = contrastive_conceptor(C_pos, C_neg)

        conceptors[task] = C_steer
        logger.info(f"  C_steer quota = {np.trace(C_steer):.1f}")

    return conceptors


def compute_similarity_matrix(conceptors, task_order):
    """Compute pairwise Frobenius cosine similarity between conceptors."""
    n = len(task_order)
    sim = np.zeros((n, n))
    for i in range(n):
        Ci = conceptors[task_order[i]]
        norm_i = np.linalg.norm(Ci, 'fro')
        for j in range(i, n):
            Cj = conceptors[task_order[j]]
            norm_j = np.linalg.norm(Cj, 'fro')
            s = np.trace(Ci.T @ Cj) / (norm_i * norm_j + 1e-10)
            sim[i, j] = s
            sim[j, i] = s
    return sim


def plot_conceptor_similarity_clustermap(conceptors, layer, output_dir):
    """Panel A: Clustered similarity heatmap with dendrograms."""
    apply_neurips_style()
    out_dir = output_dir

    task_order = sorted(conceptors.keys())
    sim_matrix = compute_similarity_matrix(conceptors, task_order)

    # Convert to distance and compute linkage
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0, None)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='ward')
    ordering = leaves_list(Z)

    # Clean task labels
    labels = [t.replace('-v3', '') for t in task_order]

    # Plot clustermap
    g = sns.clustermap(
        sim_matrix,
        row_linkage=Z,
        col_linkage=Z,
        xticklabels=labels,
        yticklabels=labels,
        cmap='mako',
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor='white',
        figsize=(8, 8),
        dendrogram_ratio=(0.12, 0.12),
        cbar_kws={'label': 'Conceptor similarity (Frobenius)', 'shrink': 0.6},
    )

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                  fontsize=7, rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                  fontsize=7)

    g.savefig(out_dir / "analysis_6_clustermap.pdf", bbox_inches='tight', pad_inches=0.05)
    g.savefig(out_dir / "analysis_6_clustermap.png", bbox_inches='tight', pad_inches=0.05,
              dpi=300)
    plt.close('all')
    logger.info("Saved analysis_6_clustermap.pdf/png")

    return sim_matrix, task_order, Z, ordering


def plot_conceptor_umap(conceptors, category_map, layer, output_dir):
    """Panel B: 2D embedding colored by semantic category (MDS on precomputed distances)."""
    apply_neurips_style()
    from sklearn.manifold import MDS
    out_dir = output_dir

    task_order = sorted(conceptors.keys())
    sim_matrix = compute_similarity_matrix(conceptors, task_order)
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0, None)

    # MDS embedding (precomputed dissimilarity)
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=42,
        n_init=10,
        max_iter=1000,
        normalized_stress='auto',
    )
    embedding = mds.fit_transform(dist_matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    categories = [category_map.get(t, "other") for t in task_order]
    unique_cats = sorted(set(categories))

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        idx = [i for i, m in enumerate(mask) if m]
        color = CATEGORY_COLORS.get(cat, '#888888')
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=color, s=80, alpha=0.85,
                   edgecolors='white', linewidth=0.6,
                   label=cat, zorder=3)

    # Label points
    try:
        from adjustText import adjust_text
        texts = []
        for i, task in enumerate(task_order):
            label = task.replace('-v3', '')
            texts.append(ax.text(embedding[i, 0], embedding[i, 1], label,
                                 fontsize=7, ha='center', va='center'))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray',
                                            lw=0.4, alpha=0.5))
    except ImportError:
        for i, task in enumerate(task_order):
            label = task.replace('-v3', '')
            ax.annotate(label, (embedding[i, 0], embedding[i, 1]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.9, edgecolor='lightgray')

    fig.savefig(out_dir / "analysis_6_umap.pdf", bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_dir / "analysis_6_umap.png", bbox_inches='tight', pad_inches=0.05,
                dpi=300)
    plt.close(fig)
    logger.info("Saved analysis_6_umap.pdf/png")

    return embedding


def make_task_space_figure(conceptors, category_map, layer, output_dir):
    """Generate both panels and a combined figure."""

    # Panel A: clustermap (saved standalone)
    sim_matrix, task_order, Z, ordering = plot_conceptor_similarity_clustermap(
        conceptors, layer, output_dir)

    # Panel B: UMAP (saved standalone)
    embedding = plot_conceptor_umap(conceptors, category_map, layer, output_dir)

    # Save data
    data = {
        "task_order": task_order,
        "similarity_matrix": sim_matrix.tolist(),
        "umap_embedding": embedding.tolist(),
        "categories": {t: category_map.get(t, "other") for t in task_order},
    }
    with open(output_dir / "analysis_6_data.json", "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved analysis_6_data.json")

    return sim_matrix, task_order, Z, ordering, embedding


def run_analysis_6():
    """Main entry point: compute conceptors and generate task space figure."""
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("=" * 70)
    logger.info("ANALYSIS 6: Task Space Structure (Similarity + MDS)")
    logger.info("=" * 70)

    cache_path = out_dir / "analysis_6_conceptors.npz"

    if cache_path.exists():
        logger.info(f"Loading cached conceptors from {cache_path}")
        data = np.load(cache_path)
        conceptors = {k: data[k] for k in data.files}
        logger.info(f"Loaded {len(conceptors)} conceptors from cache")
    else:
        # Step 1: Find steerable tasks
        logger.info("Finding steerable tasks...")
        steerable = find_steerable_tasks(MIXED_TASKS)
        logger.info(f"Found {len(steerable)} steerable tasks")

        # Step 2: Compute contrastive conceptors for all tasks
        logger.info("Computing contrastive conceptors for all tasks...")
        conceptors = compute_all_conceptors(steerable, layer_idx=2)

        # Cache to disk
        logger.info(f"Caching conceptors to {cache_path}")
        np.savez_compressed(cache_path, **conceptors)

    # Step 3: Generate figures
    logger.info("Generating task space figures...")
    make_task_space_figure(conceptors, CATEGORY_MAP, layer=11, output_dir=out_dir)

    logger.info("Analysis 6 complete.")
    return conceptors


if __name__ == "__main__":
    run_analysis_6()
