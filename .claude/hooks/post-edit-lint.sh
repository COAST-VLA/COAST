#!/bin/bash
# Post-edit hook: Run ruff check on edited Python files to catch issues early.

INPUT=$(cat)

json_field() {
  JSON_INPUT="$INPUT" JSON_PATH="$1" python3 - <<'PY'
import json
import os

try:
    value = json.loads(os.environ.get("JSON_INPUT", "{}"))
except json.JSONDecodeError:
    value = {}

for key in os.environ["JSON_PATH"].split("."):
    if not isinstance(value, dict):
        value = ""
        break
    value = value.get(key, "")

print("" if value is None else value)
PY
}

FILE_PATH=$(json_field "tool_input.file_path")

# Only check Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Skip files in excluded directories (matching ruff config in pyproject.toml)
if echo "$FILE_PATH" | grep -qE '(third_party/|\.claude/|docker/|transformers_replace/)'; then
  exit 0
fi

# Check if file still exists (might have been deleted)
if [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

PROJECT_DIR=$(json_field "cwd")
if [ -z "$PROJECT_DIR" ]; then
  PROJECT_DIR="$CLAUDE_PROJECT_DIR"
fi

cd "$PROJECT_DIR" || exit 0

# Run ruff check (report only, don't fix — just surface issues)
OUTPUT=$(uv run ruff check "$FILE_PATH" 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] && [ -n "$OUTPUT" ]; then
  # Count issues
  ISSUE_COUNT=$(echo "$OUTPUT" | grep -c "^")
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"Ruff found ${ISSUE_COUNT} issue(s) in ${FILE_PATH}:\n${OUTPUT}\"}}"
fi

exit 0
