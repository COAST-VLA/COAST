#!/bin/bash
# Post-edit hook: Run ruff check on edited Python files to catch issues early.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

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

PROJECT_DIR=$(echo "$INPUT" | jq -r '.cwd // empty')
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
