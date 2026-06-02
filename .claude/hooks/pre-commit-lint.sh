#!/bin/bash
# Pre-commit hook: Auto-run ruff check --fix and ruff format before git commits
# so pre-commit hooks never block the commit.

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

COMMAND=$(json_field "tool_input.command")

# Only act on git commit commands
if ! echo "$COMMAND" | grep -qE '^\s*git\s+commit'; then
  exit 0
fi

PROJECT_DIR=$(json_field "cwd")
if [ -z "$PROJECT_DIR" ]; then
  PROJECT_DIR="$CLAUDE_PROJECT_DIR"
fi

cd "$PROJECT_DIR" || exit 0

# Get staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
if [ -z "$STAGED_FILES" ]; then
  exit 0
fi

# Run ruff check --fix and format on staged files
echo "$STAGED_FILES" | xargs uv run ruff check --fix --quiet 2>/dev/null
echo "$STAGED_FILES" | xargs uv run ruff format --quiet 2>/dev/null

# Re-stage any files that were modified by ruff
echo "$STAGED_FILES" | xargs git add

exit 0
