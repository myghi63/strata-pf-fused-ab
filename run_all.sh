#!/usr/bin/env bash
# STRATA_PF_FUSED A/B on one card: arm 1 (fused on) then arm 0 (fused off).
#
#   bash run_all.sh --yes            # stops the running server, runs both arms, leaves it stopped
#   bash run_all.sh --yes --restore  # ... and starts the preset again at the end
#
# Everything in the preset stays as it is. Only STRATA_PF_FUSED changes between the arms.
#
# For the duration of the run the server's reasoning-loop detector is switched on
# ("reasoning_loop_recovery": "stop" in the model's installed config, the file the launch
# copies its settings from). It is off by default; with it on, a reasoning loop is cut and
# logged instead of running to max_tokens. The original file is restored on exit (also on
# Ctrl-C or an error). LOOP_RECOVERY=off leaves the file alone.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${PRESET:-iq3_s-256k-final}"
STRATA="${STRATA:-$(command -v strata || echo "$HOME/.local/bin/strata")}"
LOOP_RECOVERY="${LOOP_RECOVERY:-stop}"
CLIENT_ARGS="${CLIENT_ARGS:-}"
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
disconnected for the duration (about 3-4 hours on an RX 7800 XT).

It also sets "reasoning_loop_recovery": "$LOOP_RECOVERY" in the model's installed
config for the duration and restores the file afterwards.

Re-run with:  bash $0 --yes
EOF
  exit 1
fi

[ -x "$STRATA" ] || { echo "strata not found at $STRATA" >&2; exit 2; }
python3 -c "import numpy" 2>/dev/null || { echo "python3 with numpy is required" >&2; exit 2; }
PRESET_FILE="$HOME/.config/strata/launch-presets/$PRESET.json"
[ -f "$PRESET_FILE" ] || {
  echo "preset not found: $PRESET_FILE" >&2
  echo "set PRESET=<name> to use another one" >&2
  exit 2
}
mkdir -p "$DIR/dumps" "$DIR/out"
rm -rf "$DIR"/dumps/fused1.* "$DIR"/dumps/fused0.* "$DIR"/out/arm1 "$DIR"/out/arm0

# The engine log path is recorded in the run config; fall back to the usual name.
CFG="$HOME/.local/state/strata/launch/$PRESET.json"
if [ -f "$CFG" ]; then
  LOG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("log") or "")' "$CFG" 2>/dev/null || true)
  [ -n "$LOG" ] || LOG="$HOME/.local/state/strata/launch/${PRESET// /_}.engine.log"
fi

# The model's installed config (the launch copies its keys into the run config at every start):
# the one whose model_name matches the run config's, or MODEL_CONFIG.
MODEL_CONFIG="${MODEL_CONFIG:-}"
if [ -z "$MODEL_CONFIG" ] && [ -f "$CFG" ]; then
  MODEL_CONFIG=$(python3 - "$CFG" <<'PY' || true
import glob, json, os, sys
name = json.load(open(sys.argv[1])).get("model_name")
for p in sorted(glob.glob(os.path.expanduser("~/.local/share/strata/install/strata-*.json"))):
    try:
        if json.load(open(p)).get("model_name") == name:
            print(p); break
    except Exception:
        pass
PY
)
fi
BACKUP=""
restore_config() {
  if [ -n "$BACKUP" ] && [ -f "$BACKUP" ]; then
    cp -p "$BACKUP" "$MODEL_CONFIG" && rm -f "$BACKUP"
    echo "== restored $MODEL_CONFIG"
    BACKUP=""
  fi
}
trap restore_config EXIT
trap 'restore_config; exit 130' INT TERM
if [ "$LOOP_RECOVERY" != "off" ]; then
  [ -n "$MODEL_CONFIG" ] && [ -f "$MODEL_CONFIG" ] || {
    echo "cannot find the model's installed config (set MODEL_CONFIG=<path>, or LOOP_RECOVERY=off)" >&2
    exit 2
  }
  BACKUP="$DIR/.model-config.backup.json"
  cp -p "$MODEL_CONFIG" "$BACKUP"
  python3 - "$MODEL_CONFIG" "$LOOP_RECOVERY" <<'PY'
import json, sys
p, mode = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["reasoning_loop_recovery"] = mode
open(p, "w").write(json.dumps(d, indent=1) + "\n")
PY
  echo "== reasoning_loop_recovery=$LOOP_RECOVERY in $MODEL_CONFIG (restored at the end)"
fi

python3 "$DIR/longctx.py" fetch >/dev/null || { echo "could not download the long-context text" >&2; exit 2; }

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
  if [ "$LOOP_RECOVERY" != "off" ]; then
    python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get("reasoning_loop_recovery"); print("-- run config reasoning_loop_recovery:", v); sys.exit(v != sys.argv[2])' \
      "$CFG" "$LOOP_RECOVERY" || { echo "the launch did not carry reasoning_loop_recovery into $CFG" >&2; exit 1; }
  fi

  # shellcheck disable=SC2086
  python3 "$DIR/client.py" --arm "$ARM" --preset "$PRESET" --dumps "$DIR/dumps/fused$ARM" \
      --server-log "$SLOG" $CLIENT_ARGS

  tail -c +$((off + 1)) "$LOG" > "$DIR/arm$ARM.engine.log" 2>/dev/null || true
  tail -c +$((soff + 1)) "$SLOG" > "$DIR/arm$ARM.server.log" 2>/dev/null || true
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

restore_config

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
