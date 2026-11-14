#!/usr/bin/env bash
# Tool-name margin A/B for STRATA_PF_FUSED: arm 1 (fused) then arm 0 (MMQ).
#
#   bash run_agentic.sh --yes
#
# The probe asks the model (thinking off) to write the tool call as one line of
# JSON, so the name token is normal content and the server's logprobs give its
# distribution.  It also sends the same prompts through the real tool-call path.
# Everything in the preset stays as it is; only STRATA_PF_FUSED changes.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${PRESET:-iq3_s-256k-final}"
STRATA="${STRATA:-$(command -v strata || echo "$HOME/.local/bin/strata")}"
OUT="$DIR/agentic/out"
YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ "$YES" -eq 1 ] || { echo "this stops the running server twice; re-run with --yes"; exit 0; }
[ -x "$STRATA" ] || { echo "strata not found at $STRATA" >&2; exit 2; }

mkdir -p "$DIR/agentic"
rm -rf "$OUT"
rm -f "$DIR"/agentic/arm*.launch.txt

restore() {
  echo "== starting $PRESET again"
  "$STRATA" launch --preset "$PRESET" --no-tui --yes >/dev/null 2>&1 || true
}
trap restore EXIT

for ARM in 1 0; do
  echo
  echo "================ arm $ARM (STRATA_PF_FUSED=$ARM) ================"
  "$STRATA" launch --stop --preset "$PRESET" >/dev/null 2>&1 || true
  sleep 2
  "$STRATA" launch --preset "$PRESET" \
      --set "STRATA_PF_FUSED=$ARM" --no-tui --yes --restart \
      | tee "$DIR/agentic/arm$ARM.launch.txt"
  grep -qE "Strata is serving|Already serving" "$DIR/agentic/arm$ARM.launch.txt" || {
    echo "the server did not start (see agentic/arm$ARM.launch.txt)" >&2
    exit 1
  }
  python3 "$DIR/agentic.py" --arm "$ARM" --preset "$PRESET" --out "$OUT"
  python3 "$DIR/agentic.py" --arm "$ARM" --preset "$PRESET" --out "$OUT" --syntax 8
  "$STRATA" launch --stop --preset "$PRESET" >/dev/null 2>&1 || true
  sleep 2
done

echo
echo "================ analysis ================"
python3 "$DIR/agentic.py" --analyze "$OUT" | tee "$DIR/agentic/report.txt"
