#!/usr/bin/env bash
# Runs the pre-registered holdout ONCE, the moment gb/flat 2016 finishes.
#
# Driven by cron because the scrape spans days and reboots: a session ending
# should not delay the verdict. Idempotent — the result file is the lock, so
# this can fire every 30 minutes and will only ever run once.
#
# Nothing here chooses anything: holdout.py's protocol was committed on
# 2026-09-16 and it refuses a partial year on its own.
set -u
REPO=/home/seanleishman/Projects/gbp-horse-prediction
RESULT=$REPO/holdout_result.txt
SCRAPE_LOG=$HOME/Projects/rpscrape-community/scrape2016.log

[ -f "$RESULT" ] && exit 0
grep -q "ALL 2016 DONE" "$SCRAPE_LOG" 2>/dev/null || exit 0

cd "$REPO" || exit 1
TMP=$(mktemp)
{
  echo "=== holdout run $(date -Is) ==="
  echo "--- rebuilding dataset (both scraper trees)"
  .venv/bin/python data_import.py 2>&1 | grep -viE "warning" | tail -3
  echo "--- holdout.py (transformer seeds 0,1,2; threshold fixed 3%)"
  .venv/bin/python holdout.py 2>&1 | grep -viE "warning|stage1 epoch"
  echo "=== finished $(date -Is) ==="
} > "$TMP" 2>&1

# Only a RUN counts as the result. holdout.py exits non-zero when the year is
# short, and promoting that refusal would make the lock file permanent and burn
# the single run on nothing. Keep failures in a separate log and try again.
if grep -q "PORTFOLIO" "$TMP"; then
  mv "$TMP" "$RESULT"
else
  { echo "--- holdout not run $(date -Is):"; cat "$TMP"; } >> "$REPO/holdout_attempts.log"
  rm -f "$TMP"
fi
