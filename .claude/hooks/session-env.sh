#!/bin/bash
# SessionStart hook: persist environment variables for the session.

if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo 'export MUJOCO_GL=egl' >> "$CLAUDE_ENV_FILE"
  echo 'export GIT_LFS_SKIP_SMUDGE=1' >> "$CLAUDE_ENV_FILE"
fi
exit 0
