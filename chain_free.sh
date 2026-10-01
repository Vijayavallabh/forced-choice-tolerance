#!/bin/bash
# Wait for this card's grid sweep to finish, then run free-response on MMLU-Pro's
# ten-option file. Chained on the sweep's own "done" line rather than on the
# card being idle, because a waiter that polls an idle card slips into the gap
# between two arms and collides.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"; LOG="$2"; shift 2
# Bounded, for the reason run_agentic14.sh spells out: the 2>/dev/null that
# lets the marker arrive late also hides a wrong path and a dead producer.
[ -n "$LOG" ] || { echo "chain_free: no log to wait on" >&2; exit 2; }
[ -d "$(dirname "$LOG")" ] || {
  echo "chain_free: $(dirname "$LOG") does not exist, so $LOG never will" >&2
  exit 2; }
waited=0; limit=${WAIT_TIMEOUT:-21600}
until grep -q "^done bixgrid" "$LOG" 2>/dev/null; do
  sleep 30
  waited=$((waited + 30))
  [ "$waited" -lt "$limit" ] || {
    echo "chain_free: $LOG has no 'done bixgrid' after ${waited}s" >&2; exit 3; }
done
DEV="$DEV" ITEMS=build/mmlu_pro_matched_released.jsonl TAG=mmlupro \
  OUT=results/free_response_mmlupro.json ./run_free_response.sh "$@"
