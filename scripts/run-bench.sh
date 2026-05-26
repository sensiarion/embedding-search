#!/usr/bin/env bash
# Detached bench sweep runner. nohup + setsid so the job survives this
# shell exiting. Writes to bench-<utc-iso>.log and prints the PID to
# stdout. The caller then polls the log for the completion marker
# "BENCH DONE" (printed at the very end on success or failure).
set -euo pipefail

cd "$(dirname "$0")/.."

ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="benchmarks/results/bench-${ts}.log"
mkdir -p benchmarks/results

models="${MODELS:-sensiarion/CodeRankEmbed-f16,minishlab/potion-base-32M,minishlab/potion-multilingual-128M,google/embeddinggemma-300m}"

# Wrap the cargo invocation so we can append a single, greppable
# completion marker regardless of how the inner command exits.
inner="cargo xtask golden --models '${models}'; ec=\$?; echo \"BENCH DONE rc=\$ec at \$(date -u +%FT%TZ)\"; exit \$ec"

# macOS has no setsid; nohup + disown is enough to survive shell exit.
# stdin from /dev/null ensures no SIGTTIN on background read.
nohup bash -c "$inner" </dev/null >"$log" 2>&1 &
pid=$!
disown

# Print machine-parseable summary on a single line, then exit so the
# caller can detach and poll independently.
echo "PID=$pid LOG=$log STARTED=$(date -u +%FT%TZ)"
