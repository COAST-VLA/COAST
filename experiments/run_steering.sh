#!/bin/bash
#SBATCH --job-name=conceptor-steer
#SBATCH --partition=p_nlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --constraint=48GBgpu
#SBATCH --output=experiments/steering_results/slurm_%j.out
#SBATCH --error=experiments/steering_results/slurm_%j.err

# Conceptor Steering Experiment for π₀.₅
# Runs Strategy 3 (global), Strategy 5 (per-step), linear, and random baselines

set -e

PYTHON=/nlpgpu/data/miaom/openpi-metaworld/.venv/bin/python
export MUJOCO_GL=osmesa
export HF_HOME=/nlp/data/huggingface_cache
export TORCH_COMPILE_DISABLE=1

cd /nlpgpu/data/miaom/openpi-metaworld

mkdir -p experiments/steering_results

$PYTHON experiments/conceptor_steering.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --tasks assembly-v3 \
    --alphas 0.1 0.5 1.0 \
    --betas 0.1 0.3 0.5 \
    --steering-layer 11 \
    --linear-alphas 0.5 1.0 2.0 5.0 \
    --num-envs 5 \
    --max-steps 300 \
    --replan-steps 10 \
    --output-dir experiments/steering_results
