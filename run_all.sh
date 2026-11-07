#!/usr/bin/env bash
# STRATA_PF_FUSED A/B on one card: arm 1 (fused on) then arm 0 (fused off).
#
#   bash run_all.sh --yes            # stops the running server, runs both arms, leaves it stopped
#   bash run_all.sh --yes --restore  # ... and starts the preset again at the end
#
# Everything else (preset, context, drafting, conversation cache, --ple-io) stays
# exactly as it is in the preset. Only STRATA_PF_FUSED changes between the arms.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${PRESET:-iq3_s-256k-final}"
STRATA="${STRATA:-$(command -v strata || echo "$HOME/.local/bin/strata")}"
LOG="$HOME/.local/state/strata/launch/${PRESET// /_}.engine.log"
SLOG="$HOME/.local/state/strata/runtime/launch-${PRESET// /_}.server.log"
YES=0
RESTORE=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --restore) RESTORE=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$YES" -ne 1 ]; then
  cat <<EOF
This stops the Strata server that is running now (if any), starts it twice
(fused on / fused off) and stops it at the end. Your chat clients will be
disconnected for the duration.

Re-run with:  bash $0 --yes
EOF
  exit 1
fi

[ -x "$STRATA" ] || { echo "strata not found at $STRATA" >&2; exit 2; }
python3 -c "import numpy" 2>/dev/null || { echo "python3 with numpy is required" >&2; exit 2; }
[ -f "$HOME/.config/strata/launch-presets/$PRESET.json" ] || {
  echo "preset not found: $HOME/.config/strata/launch-presets/$PRESET.json" >&2
  echo "set PRESET=<name> to use another one" >&2
  exit 2
}
mkdir -p "$DIR/dumps" "$DIR/out"
rm -f "$DIR"/dumps/fused1.* "$DIR"/dumps/fused0.*

# The engine log path is recorded in the run config; fall back to the usual name.
CFG="$HOME/.local/state/strata/launch/$PRESET.json"
if [ -f "$CFG" ]; then
  LOG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("log") or "")' "$CFG" 2>/dev/null || true)
  [ -n "$LOG" ] || LOG="$HOME/.local/state/strata/launch/${PRESET// /_}.engine.log"
fi

echo "== preset: $PRESET"
echo "== strata: $STRATA"

for ARM in 1 0; do
  echo
  echo "================ arm $ARM (STRATA_PF_FUSED=$ARM) ================"
  "$STRATA" launch --stop --preset "$PRESET" >/dev/null 2>&1 || true
  sleep 2
  off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  soff=$(stat -c %s "$SLOG" 2>/dev/null || echo 0)

  "$STRATA" launch --preset "$PRESET" \
      --set "STRATA_PF_FUSED=$ARM" \
      --set "STRATA_DUMP_FIRST_LOGITS=$DIR/dumps/fused$ARM" \
      --no-tui --yes --restart | tee "$DIR/arm$ARM.launch.txt"
  grep -qE "Strata is serving|Already serving" "$DIR/arm$ARM.launch.txt" || {
    echo "the server did not start (see $DIR/arm$ARM.launch.txt)" >&2
    exit 1
  }

  python3 "$DIR/client.py" --arm "$ARM" --preset "$PRESET" --dumps "$DIR/dumps/fused$ARM"

  tail -c +$((off + 1)) "$LOG" > "$DIR/arm$ARM.engine.log" 2>/dev/null || true
  tail -c +$((soff + 1)) "$SLOG" > "$DIR/arm$ARM.server.log" 2>/dev/null || true
  grep -c "the thinking repeated the same passages" "$DIR/arm$ARM.server.log" > "$DIR/arm$ARM.server_loops" 2>/dev/null || echo 0 > "$DIR/arm$ARM.server_loops"
  fused=$(grep -c "prompt experts on the fused int8 kernels" "$DIR/arm$ARM.engine.log" || true)
  echo "-- engine log: $DIR/arm$ARM.engine.log"
  echo "-- fused-expert marker lines: $fused (arm1 should be >=1, arm0 should be 0)"
  python3 - "$DIR" "$ARM" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]) / "out" / f"arm{sys.argv[2]}" / "kl"
ns = []
for p in sorted(d.glob("*.json")):
    t = (json.loads(p.read_text()).get("response") or {}).get("timings") or {}
    if t.get("prompt_n"):
        ns.append(int(t["prompt_n"]))
if ns:
    print(f"-- KL prompt tokens: min {min(ns)}, max {max(ns)} "
          f"(the fused prompt path needs a streamed chunk >= 1024 tokens)")
PY
  if [ "$ARM" = "1" ] && [ "$fused" = "0" ]; then
    echo "WARNING: arm1 never logged the fused prompt experts. If the prompts are below ~1024" >&2
    echo "         tokens, both arms run MMQ and this run measures nothing (see above)." >&2
  fi
  if [ "$ARM" = "0" ] && [ "$fused" != "0" ]; then
    echo "WARNING: arm0 still logged the fused prompt experts; STRATA_PF_FUSED=0 did not disable it." >&2
  fi

  "$STRATA" launch --stop --preset "$PRESET" >/dev/null 2>&1 || true
  sleep 2
done

echo
echo "================ analysis ================"
python3 "$DIR/report.py" | tee "$DIR/report.txt" || true
python3 "$DIR/kl.py" | tee "$DIR/kl.txt" || true

if [ "$RESTORE" -eq 1 ]; then
  echo
  echo "== starting $PRESET again"
  "$STRATA" launch --preset "$PRESET" --no-tui --yes
else
  echo
  echo "Server left stopped. Start it again with:"
  echo "  $STRATA launch --preset \"$PRESET\""
fi
