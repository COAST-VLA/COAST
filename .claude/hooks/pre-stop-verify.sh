#!/bin/bash
# Stop hook: verify ruff passes on changed files before Claude finishes responding.

PROJECT_DIR="$CLAUDE_PROJECT_DIR"
if [ -z "$PROJECT_DIR" ]; then
  exit 0
fi

cd "$PROJECT_DIR" || exit 0

# Get Python files changed vs HEAD (staged + unstaged)
CHANGED_FILES=$(git diff --name-only HEAD -- '*.py' 2>/dev/null)
if [ -z "$CHANGED_FILES" ]; then
  exit 0
fi

# Filter to files that still exist
EXISTING_FILES=""
while IFS= read -r f; do
  if [ -f "$f" ]; then
    if [ -z "$EXISTING_FILES" ]; then
      EXISTING_FILES="$f"
    else
      EXISTING_FILES="$EXISTING_FILES $f"
    fi
  fi
done <<< "$CHANGED_FILES"

if [ -z "$EXISTING_FILES" ]; then
  exit 0
fi

OUTPUT=$(uv run ruff check $EXISTING_FILES 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] && [ -n "$OUTPUT" ]; then
  ISSUE_COUNT=$(echo "$OUTPUT" | grep -c "^")
  echo "Ruff found ${ISSUE_COUNT} issue(s) in changed files. Fix them before finishing:" >&2
  echo "$OUTPUT" >&2
  exit 2
fi

FORMAT_OUTPUT=$(uv run ruff format --check $EXISTING_FILES 2>/dev/null)
FORMAT_EXIT_CODE=$?

if [ $FORMAT_EXIT_CODE -ne 0 ] && [ -n "$FORMAT_OUTPUT" ]; then
  FORMAT_ISSUE_COUNT=$(echo "$FORMAT_OUTPUT" | grep -c "^Would reformat:")
  echo "Ruff format found ${FORMAT_ISSUE_COUNT} formatting issue(s) in file(s). Run \`uv run ruff format $EXISTING_FILES\` to fix, then retry." >&2
  echo "$FORMAT_OUTPUT" >&2
  exit 2
fi

exit 0
