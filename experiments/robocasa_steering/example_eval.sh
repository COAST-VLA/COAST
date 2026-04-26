#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%j_generate_monolithic_train56_seed4.out
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/generate_data.py \
    --base_data_path /home/spatank/compositional-rl-synth-data/data \
    --base_results_folder /home/spatank/compositional-rl-synth-data/results/diffusion \
    --gin_config_files /home/spatank/compositional-rl-synth-data/config/diffusion.gin \
    --denoiser monolithic \
    --task_list_path /home/spatank/compositional-rl-synth-data/offline_compositional_rl_datasets/_train_test_splits \
    --num_train 56 \
    --dataset_type expert \
    --experiment_type default \
    --seed 0 \
    --run 1