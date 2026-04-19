#!/bin/bash
# PostCompact hook: re-inject session state after context compaction.
# Skills write to .claude/.session_state before launching long operations.

STATE_FILE="$CLAUDE_PROJECT_DIR/.claude/.session_state"
if [ -f "$STATE_FILE" ]; then
  STATE=$(cat "$STATE_FILE" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostCompact\",\"additionalContext\":\"[SESSION STATE restored after compaction]\\n${STATE}\"}}"
fi
exit 0
