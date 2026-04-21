#!/usr/bin/env bash
# Upload an activation dataset to the HuggingFace Hub one task at a time.
#
# HF caps each commit at 25k files. Activation datasets often exceed that
# (a single `hf upload <root>` fails with "413 Payload Too Large"), so this
# splits the upload into one commit per task.
#
# Expected local layout:
#   <local_root>/<step>/<task>/...
# The script uploads each <task> dir to path_in_repo=<step>/<task>.
#
# Usage:
#   scripts/upload_activations_by_task.sh <repo_id> <local_root> [step]
#
# If [step] is omitted, every immediate subdir of <local_root> is treated
# as a step and uploaded.
#
# Examples:
#   # Upload one step's worth of collected data for an existing dataset repo:
#   scripts/upload_activations_by_task.sh \
#       brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env \
#       activations/pi0fast-metaworld-activations-v1-ml45train-16env \
#       2500
#
#   # Upload every step dir under <local_root> (omit the 3rd arg):
#   scripts/upload_activations_by_task.sh \
#       brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env \
#       activations/pi0fast-metaworld-activations-v1-ml45train-16env

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <repo_id> <local_root> [step]" >&2
    exit 1
fi

repo_id="$1"
local_root="$2"
step_filter="${3:-}"

if [[ ! -d "$local_root" ]]; then
    echo "error: local_root '$local_root' is not a directory" >&2
    exit 1
fi

# Ensure dataset repo exists (idempotent).
hf repo create "$repo_id" --repo-type dataset --exist-ok >/dev/null

upload_task() {
    local step="$1"
    local task_dir="$2"
    local task
    task="$(basename "$task_dir")"
    local path_in_repo="${step}/${task}"

    echo ">>> uploading ${path_in_repo}  (from ${task_dir})"
    hf upload "$repo_id" "$task_dir" "$path_in_repo" \
        --repo-type dataset \
        --commit-message "Upload ${path_in_repo}"
}

if [[ -n "$step_filter" ]]; then
    steps=("$step_filter")
else
    steps=()
    while IFS= read -r -d '' d; do
        steps+=("$(basename "$d")")
    done < <(find "$local_root" -mindepth 1 -maxdepth 1 -type d -print0)
fi

for step in "${steps[@]}"; do
    step_dir="${local_root%/}/${step}"
    if [[ ! -d "$step_dir" ]]; then
        echo "warn: skipping '$step_dir' (not a directory)" >&2
        continue
    fi
    while IFS= read -r -d '' task_dir; do
        upload_task "$step" "$task_dir"
    done < <(find "$step_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "Done. View at https://huggingface.co/datasets/${repo_id}"
