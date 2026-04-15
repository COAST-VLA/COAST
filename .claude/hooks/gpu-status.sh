#!/bin/bash
# UserPromptSubmit hook: inject current GPU status into every prompt.
# Claude always has up-to-date GPU info without needing to run nvidia-smi.

if ! command -v nvidia-smi &>/dev/null; then
  echo '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[GPU STATUS] nvidia-smi unavailable."}}'
  exit 0
fi

GPU_CSV=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$GPU_CSV" ]; then
  echo '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[GPU STATUS] nvidia-smi query failed."}}'
  exit 0
fi

FREE_IDS=""
while IFS=',' read -r idx mem_used mem_total gpu_util; do
  idx=$(echo "$idx" | xargs)
  mem_used=$(echo "$mem_used" | xargs)
  if [ "$mem_used" -lt 1000 ]; then
    if [ -z "$FREE_IDS" ]; then
      FREE_IDS="$idx"
    else
      FREE_IDS="$FREE_IDS,$idx"
    fi
  fi
done <<< "$GPU_CSV"

if [ -z "$FREE_IDS" ]; then
  FREE_IDS="NONE"
fi

# Escape newlines for JSON
GPU_TABLE=$(echo "$GPU_CSV" | sed 's/$/\\n/' | tr -d '\n')

echo "{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"[GPU STATUS] Free GPUs (< 1GB used): ${FREE_IDS}\\nindex, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\\n${GPU_TABLE}\"}}"
exit 0
