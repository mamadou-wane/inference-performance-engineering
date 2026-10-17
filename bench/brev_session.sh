#!/usr/bin/env bash
# Run one bench/session.sh stage on a new Brev instance's host, then delete the instance, however the
# stage ends:  bench/brev_session.sh TYPE SESSION_BUDGET_USD SESSION_MINUTES COMMIT STAGE RESULTS_DIR
# Budget and minutes are session totals; RESULTS_DIR/brev-ledger.tsv records each run at whole hours.
# Results arrive as RESULTS_DIR/<instance>/ after a SHA-256 check. Exit 3: deletion not confirmed.
# Not covered: SIGKILL, the machine sleeping or losing power, the Brev organization changing mid-run.
set -uo pipefail
set -m        # each background job gets its own process group, so a kill reaches its children
trap '' PIPE  # output piped to a reader that exits must not stop the cleanup
[ $# -eq 6 ] || { sed -n 2,6p "$0"; exit 2; }
type=$1 budget=$2 minutes=$3 commit=$4 stage=$5 results=$6
case $stage in screen | probe | long | bench) ;; *) echo "unknown stage $stage"; exit 2 ;; esac
name=ipe-$stage-$(date +%s)-$$ status=1 gone=0 started= created=0 active=
margin=${IPE_DELETE_MARGIN_S:-600} poll=${IPE_POLL_S:-15} copy_s=${IPE_COPY_S:-120} call_s=${IPE_CALL_S:-60}
[ "$margin" -ge $((call_s + copy_s + 20 * poll)) ] || { echo "IPE_DELETE_MARGIN_S is too small"; exit 2; }
ledger=$results/brev-ledger.tsv lock=$results/.brev-session.lock
mkdir -p "$results" || exit 2
mkdir "$lock" 2>/dev/null || { echo "$lock exists: another run is active, or remove it"; exit 2; }
trap 'rmdir "$lock"' EXIT
trap 'exit 130' INT TERM HUP
[ -s "$ledger" ] || printf 'instance\ttype\tseconds\tprice_per_hour\tworst_case_usd\tdeleted\n' > "$ledger"

say() {  # external echo: a failed write leaves nothing in this shell's buffers
  /bin/echo "$name: $*"
  /bin/echo "$name: $*" >> "$results/brev-session.log"
}

bounded() {  # DEADLINE COMMAND...: run COMMAND in its own process group until SECONDS reaches DEADLINE
  local deadline=$1 pid result
  shift
  "$@" < /dev/null &
  pid=$! active=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then kill -- -"$pid" 2>/dev/null; wait "$pid"; active=; return 124; fi
    sleep 0.2
  done
  wait "$pid"
  result=$? active=
  return $result
}

sha256() { python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

listed() {  # 0: the instance is listed; 1: it is not; 2: no usable listing
  local listing
  listing=$(bounded $((SECONDS + call_s)) brev ls --json) || return 2
  python3 -c '
import json, sys
try:
    workspaces = json.loads(sys.stdin.read())["workspaces"] or []
except (ValueError, KeyError, TypeError):
    sys.exit(2)
sys.exit(0 if any(w.get("name") == sys.argv[1] for w in workspaces) else 1)' "$name" <<< "$listing"
}

price=$(bounded $((SECONDS + call_s)) brev search --json | python3 -c '
import json, sys
rows = [r for r in json.load(sys.stdin) if r["type"] == sys.argv[1]]
if rows and "disk_price_per_gb_mo" not in rows[0]:  # a separately billed disk is not in price_per_hour
    print(rows[0]["price_per_hour"])' "$type")
[ -n "$price" ] || { echo "type $type is not offered now, or bills its disk separately"; exit 2; }
limit=$(python3 - "$ledger" "$budget" "$minutes" "$price" "$margin" <<'EOF'
import math, sys
path, budget, minutes, price, margin = sys.argv[1], *map(float, sys.argv[2:])
lines = open(path).read().splitlines()
if lines[0].split("\t") != ["instance", "type", "seconds", "price_per_hour", "worst_case_usd", "deleted"]:
    sys.exit(f"{path}: unexpected header")
spent_s = spent_usd = 0.0
for line in lines[1:]:
    fields = line.split("\t")
    try:
        seconds, cost = float(fields[2]), float(fields[4])
        valid = len(fields) == 6 and fields[5] in ("0", "1") and min(seconds, cost) >= 0 and math.isfinite(seconds + cost)
    except (IndexError, ValueError):
        valid = False
    if not valid:
        sys.exit(f"{path}: malformed row {line!r}")
    if fields[5] == "0":
        sys.exit(f"{path}: {fields[0]} was not confirmed deleted; check brev ls, then set its last field to 1")
    spent_s, spent_usd = spent_s + seconds, spent_usd + cost
print(int(min(minutes * 60 - spent_s, int((budget - spent_usd) / price) * 3600) - margin))
EOF
) || exit 2
[ "$limit" -gt 0 ] || { echo "the session's remaining budget or time at \$$price/h does not cover a run plus ${margin}s"; exit 2; }
say "$type at \$$price/h, stage $stage, limit ${limit}s"

retrieve() {  # one archive, accepted only if its SHA-256 matches the instance's
  local packed expected
  packed=$(bounded $((SECONDS + call_s)) brev exec "$name" --host \
    "cd ipe/results && tar czf ../$stage.tgz $stage && echo SHA256 \$(sha256sum ../$stage.tgz)") || return 1
  expected=$(awk '$1 == "SHA256" {print $2}' <<< "$packed")
  bounded $((SECONDS + copy_s)) brev copy --host "$name:ipe/$stage.tgz" "$results/$name.tgz" || return 1
  [ -n "$expected" ] && [ "$(sha256 "$results/$name.tgz")" = "$expected" ] || { say "archive hash mismatch"; return 1; }
  mkdir -p "$results/$name" && tar xzf "$results/$name.tgz" -C "$results/$name" --strip-components 1
}

cleanup() {
  trap 'say "deleting before exit"' INT TERM HUP  # a repeated Ctrl-C must not skip the deletion
  trap - EXIT
  [ -n "$active" ] && kill -- -"$active" 2>/dev/null
  if [ -n "$started" ]; then
    if retrieve; then say "results retrieved, SHA-256 verified"; else say "results not retrieved or not verified"; fi
    bounded $((SECONDS + call_s)) brev delete "$name"
  fi
  local absent=0 seconds=0 worst
  for _ in $(seq 20); do  # two usable listings without the instance; delete again while it is listed
    listed
    case $? in
      0) absent=0; bounded $((SECONDS + call_s)) brev delete "$name" ;;
      1) absent=$((absent + 1)) ;;
    esac
    [ "$absent" -ge 2 ] && { gone=1; break; }
    sleep "$poll"
  done
  [ -n "$started" ] && seconds=$((SECONDS - created))
  worst=$(python3 -c "import math; print(math.ceil(math.ceil($seconds / 3600) * $price * 100) / 100)")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$type" "$seconds" "$price" "$worst" "$gone" >> "$ledger"
  say "${seconds}s from create, at most \$$worst"
  rmdir "$lock"
  if [ "$gone" = 1 ]; then say "deleted, confirmed by brev ls"; exit "$status"; fi
  say "STILL LISTED or unconfirmed by brev ls; delete it by hand, then set its ledger row to 1"
  exit 3
}
trap 'status=130; exit' INT TERM HUP
trap cleanup EXIT

started=1 created=$SECONDS deadline=$((SECONDS + limit))
bounded "$deadline" brev create "$name" --type "$type" --timeout 600
status=$?
[ "$status" = 124 ] && say "time limit reached during create"
[ "$status" = 0 ] || exit
bounded "$deadline" brev exec "$name" --host "curl -LsSf https://astral.sh/uv/install.sh | sh && . \$HOME/.local/bin/env &&
  git clone -q https://github.com/mamadou-wane/ipe.git ipe && cd ipe && git checkout -q $commit &&
  bench/session.sh $stage results/$stage"
status=$?
[ "$status" = 124 ] && say "time limit reached"
exit
