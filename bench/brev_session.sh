#!/usr/bin/env bash
# Run one bench/session.sh stage on a new Brev instance's host, then delete the instance, however the
# stage ends:  bench/brev_session.sh TYPE SESSION_BUDGET_USD SESSION_MINUTES COMMIT STAGE RESULTS_DIR
# Budget and minutes are session totals; RESULTS_DIR/brev-ledger.tsv records each run at whole hours.
# Results arrive as RESULTS_DIR/<instance>/ after a SHA-256 check. Exit 3: deletion not confirmed; 4: no SSH.
# Not covered: SIGKILL, the machine sleeping or losing power, the Brev organization changing mid-run.
set -uo pipefail
set -m        # each background job gets its own process group, so a kill reaches its children
trap '' PIPE  # output piped to a reader that exits must not stop the cleanup
[ $# -eq 6 ] || { sed -n 2,6p "$0"; exit 2; }
type=$1 budget=$2 minutes=$3 commit=$4 stage=$5 results=$6
case $stage in screen | probe | long | bench) ;; *) echo "unknown stage $stage"; exit 2 ;; esac
name=ipe-$stage-$(date +%s)-$$ status=1 gone=0 started= created=0 active=
margin=${IPE_DELETE_MARGIN_S:-600} poll=${IPE_POLL_S:-15} copy_s=${IPE_COPY_S:-120} call_s=${IPE_CALL_S:-60}
ssh_s=${IPE_SSH_S:-900}  # how long after create the host's SSH may take to come up
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

compute=$(bounded $((SECONDS + call_s)) brev search --json | python3 -c '
import json, sys
rows = [r for r in json.load(sys.stdin) if r["type"] == sys.argv[1]]
if rows and "disk_price_per_gb_mo" not in rows[0]:  # a flexible disk of unknown size is billed separately
    print(rows[0]["price_per_hour"])' "$type")
[ -n "$compute" ] || { echo "type $type is not offered now, or bills its disk separately"; exit 2; }
# brev search drops the price of a fixed-size disk; Brev's public catalog, its source, has it. Sizes count high.
storage=$(bounded $((SECONDS + call_s)) curl -fsS -H 'Content-Type: application/json' -H 'Connect-Protocol-Version: 1' \
  -d '{"options": {"includeCpu": false, "skipAccessFilter": false}}' \
  https://api.brev.dev/devplaneapi.v1.InstanceService/ListPublicInstanceType | python3 -c '
import json, re, sys
GB = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3, "KiB": 2**10 / 1e9, "MiB": 2**20 / 1e9, "GiB": 2**30 / 1e9,
      "TiB": 2**40 / 1e9}
def gb(text):  # "128GiB" or "1TiB376GiB" in decimal GB; None if it does not parse
    parts = re.findall(r"([0-9.]+)((?:[KMGT]i?)?B)", text)
    return sum(float(n) * GB[u] for n, u in parts) if parts and "".join(n + u for n, u in parts) == text else None
rows = [i for i in json.load(sys.stdin)["items"] if i["type"] == sys.argv[1]]
if len(rows) != 1:
    sys.exit(1)
cost = 0.0
for disk in rows[0].get("supportedStorage", []):
    rate, size, size_bytes = float((disk.get("pricePerGbHr") or {}).get("amount") or 0), disk.get("size", ""), disk.get("sizeBytes") or {}
    if rate == 0 or (size == "0B" and not size_bytes.get("value")):  # unpriced, or an optional disk create does not add
        continue
    sizes = ([gb(size)] if size else []) + ([float(size_bytes["value"]) * GB[size_bytes["unit"]]] if size_bytes.get("unit") in GB else [])
    if not sizes or None in sizes:
        sys.exit(1)
    cost += max(sizes) * rate * disk.get("count", 1)
print(round(cost, 6))' "$type")
[ -n "$storage" ] || { echo "the full price of $type (compute and storage) cannot be established"; exit 2; }
price=$(python3 -c "print(round($compute + $storage, 6))")
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
say "$type at \$$price/h (\$$compute compute, \$$storage storage), stage $stage, limit ${limit}s"

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
# Brev reports Ready before the host accepts SSH, and brev exec retries for only seconds.
ssh_by=$(( created + ssh_s < deadline ? created + ssh_s : deadline ))
until bounded $(( SECONDS + call_s < ssh_by ? SECONDS + call_s : ssh_by )) brev exec "$name" --host true; do
  if [ "$SECONDS" -ge "$ssh_by" ]; then
    status=4; say "SSH to the host not available $((SECONDS - created))s after create; the stage did not run"; exit
  fi
  sleep "$poll"
done
say "SSH ready $((SECONDS - created))s after create"
bounded "$deadline" brev exec "$name" --host "curl -LsSf https://astral.sh/uv/install.sh | sh && . \$HOME/.local/bin/env &&
  git clone -q https://github.com/mamadou-wane/ipe.git ipe && cd ipe && git checkout -q $commit &&
  bench/session.sh $stage results/$stage"
status=$?
[ "$status" = 124 ] && say "time limit reached"
exit
