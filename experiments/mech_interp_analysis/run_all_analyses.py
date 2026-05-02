#!/usr/bin/env python3
"""
Run all 5 mechanistic interpretability analyses sequentially,
then assemble the master figure.

Usage:
    python experiments/mech_interp_analysis/run_all_analyses.py
"""

import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")

from shared_utils import ensure_output_dir


def main():
    ensure_output_dir()
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Subspace geometry — eigenvalue spectra")
    logger.info("=" * 70)
    from analysis_1_spectra import run_analysis_1
    run_analysis_1()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: Success/failure overlap vs performance gap")
    logger.info("=" * 70)
    from analysis_2_overlap import run_analysis_2
    run_analysis_2()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Contrastive subspace projection")
    logger.info("=" * 70)
    from analysis_3_projection import run_analysis_3
    run_analysis_3()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: Denoising step dynamics")
    logger.info("=" * 70)
    from analysis_4_dynamics import run_analysis_4
    run_analysis_4()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 5: On-manifold preservation")
    logger.info("=" * 70)
    from analysis_5_manifold import run_analysis_5
    run_analysis_5()

    logger.info("")
    logger.info("=" * 70)
    logger.info("MASTER FIGURE: Assembling 5-panel figure")
    logger.info("=" * 70)
    from master_figure import build_master_figure
    build_master_figure()

    elapsed = time.time() - t0
    logger.info(f"\nAll analyses complete in {elapsed/60:.1f} minutes.")
    logger.info(f"Results saved to: {ensure_output_dir()}")


if __name__ == "__main__":
    main()
