#!/bin/bash
# Pre-commit hook: Auto-run ruff check --fix and ruff format before git commits
# so pre-commit hooks never block the commit.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only act on git commit commands
if ! echo "$COMMAND" | grep -qE '^\s*git\s+commit'; then
  exit 0
fi

PROJECT_DIR=$(echo "$INPUT" | jq -r '.cwd // empty')
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
