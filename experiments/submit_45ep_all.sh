#!/bin/bash
# Submit all 45-episode rerun jobs for statistical significance
set -e

BASE="/vast/projects/ungar/stellar/miaom"

echo "=== Submitting pi0.5 LIBERO (10 tasks) ==="
for i in $(seq 0 9); do
    sbatch "${BASE}/openpi-new/experiments/pi05_libero/steering_results_45ep/scripts/task_${i}.sh"
done

echo "=== Submitting pi0.5 RoboCasa (7 tasks) ==="
for i in $(seq 0 6); do
    sbatch "${BASE}/openpi-new/experiments/pi05_robocasa/steering_results_45ep/scripts/task_${i}.sh"
done

echo "=== Submitting GR00T RoboCasa (7 tasks) ==="
for i in $(seq 0 6); do
    sbatch "${BASE}/openpi-groot/experiments/groot_robocasa/steering_results_45ep/scripts/task_${i}.sh"
done

echo ""
echo "=== Submitted 24 jobs total ==="
echo "Monitor with:  squeue -u \$USER"
