#!/usr/bin/env bash
set -euo pipefail

SKILL_NAME="goals-habits-tracker"
SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SKILL_ROOT/../../.." && pwd)"

SKILL_DATA_ROOT="$PROJECT_ROOT/.skills-data"
SKILL_DATA_DIR="$SKILL_DATA_ROOT/$SKILL_NAME"

mkdir -p "$SKILL_DATA_DIR/bin" "$SKILL_DATA_DIR/logs" "$SKILL_DATA_DIR/cache/pycache" "$SKILL_DATA_DIR/tmp"

ENV_FILE="$SKILL_DATA_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<EOF
SKILL_ROOT="$SKILL_ROOT"
SKILL_NAME="$SKILL_NAME"
SKILL_DATA_ROOT="$SKILL_DATA_ROOT"
SKILL_DATA_DIR="$SKILL_DATA_DIR"
GOALS_DB="$SKILL_DATA_DIR/goals.sqlite3"
WEEK_START="mon"
# Set during onboarding (IANA zone, e.g. Europe/Madrid). Empty means "unknown".
TIMEZONE=""
# How to track quit habits: occurrences (0=clean day; 1+=slips) or boolean (0/1).
QUIT_HABIT_TRACKING="occurrences"
# Optional weekly check-in preference (set during onboarding).
WEEKLY_CHECKIN_DAY=""
WEEKLY_CHECKIN_TIME=""
EOF
  echo "Wrote $ENV_FILE"
else
  echo "Exists: $ENV_FILE (leaving as-is)"
fi

CONFIG_FILE="$SKILL_DATA_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
  cat >"$CONFIG_FILE" <<'EOF'
{
  "areas": [],
  "onboarded": false,
  "onboarded_at": null
}
EOF
  echo "Wrote $CONFIG_FILE"
fi
