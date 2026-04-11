#!/bin/bash
#SBATCH --job-name=mech_a6
#SBATCH --partition=p_nlp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis/results/run_a6_gpu.out
#SBATCH --error=/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis/results/run_a6_gpu.err

export PYTHONPATH=/nlpgpu/data/miaom/torch_pkg:/nlpgpu/data/miaom/activation_inform/.packages:$PYTHONPATH
export HF_HOME=/nlp/data/huggingface_cache

PYTHON=/home1/m/miaom/miniconda3/envs/ml_env/bin/python

cd /nlpgpu/data/miaom/openpi-metaworld
$PYTHON experiments/mech_interp_analysis/analysis_6_task_space.py
