#!/bin/bash
# RecipeBench report run: everything for the midway report in under one hour of wall clock.
#   cd .../ChefBot_recipe/recipe_planning && bash bench/report_run.sh
# Hosted models run in parallel on the lite split (no GPU). Local models run one at a time on the mini split with
# the GPU duty cycle (15 s cooldown, wait above 85 C junction) and hard per-model time caps. Every run resumes from
# what is already saved, so a rerun only does what is missing.
set -u
E=/mnt/d/Jack/Documents/Stanford/Course_Work/Engineering_AI_Agents
P=/mnt/d/Jack/miniconda3/envs/chefbot/bin/python
B=$E/recipe_eval_v1/bench
O=$E/recipe_eval_v1/bench_runs
LOG=$E/logs/report_run_$(date +%Y%m%d-%H%M)
RUN="$P -m bench.run --out $O"
DL="timeout --signal=INT --kill-after=60"     # soft stop at the deadline (saved rows kept), hard kill 60 s later
mkdir -p "$LOG"
cd "$(dirname "$0")/.." || exit 1
set -a; source $E/ChefBot/.env; set +a
echo "start $(date)   logs: $LOG"

# --- hosted, in parallel. Caps are recorded spend per model (total <= $21).
( $DL 50m $RUN --instances $B/instances_lite.jsonl --model gpt-6-luna --max-cost 1.5
  $DL 25m $RUN --instances $B/instances_lite.jsonl --model gpt-6-luna --no-repair --max-cost 1
  RECIPE_REPAIR_HINTS=0 $DL 20m $RUN --instances $B/instances_lite.jsonl --model gpt-6-luna --max-cost 1 ) > $LOG/luna.log 2>&1 &
$DL 50m $RUN --instances $B/instances_lite.jsonl --model gpt-6.1-sol --max-cost 5 > $LOG/sol.log 2>&1 &
$DL 50m $RUN --instances $B/instances_lite.jsonl --model openrouter:anthropic/claude-sonnet-5.5 --max-cost 5 > $LOG/sonnet.log 2>&1 &
$DL 50m $RUN --instances $B/instances_lite.jsonl --model openrouter:anthropic/claude-haiku-5.5 --max-cost 1.5 > $LOG/haiku.log 2>&1 &
$DL 50m $RUN --instances $B/instances_lite.jsonl --model openrouter:google/gemini-3.1-pro-preview --max-cost 6 > $LOG/gemini.log 2>&1 &

# --- local, one at a time on the GPU (mini split: 14 instances each). Total local time <= 50 min.
OLLAMA_MODELS=/mnt/d/ollama-models setsid ollama serve > $LOG/ollama.log 2>&1 < /dev/null &
OLL=$!
sleep 8
$RUN --instances $B/instances_mini.jsonl --model ollama:qwen3.5:9b --reasoning-effort off --cooldown 15 --max-minutes 17 > $LOG/qwen.log 2>&1
$RUN --instances $B/instances_mini.jsonl --model ollama:qwen3.5:9b --reasoning-effort off --no-repair --cooldown 15 --max-minutes 12 > $LOG/qwen_norepair.log 2>&1
$RUN --instances $B/instances_mini.jsonl --model ollama:gpt-oss:20b --cooldown 15 --max-minutes 18 > $LOG/gptoss.log 2>&1
kill $OLL 2>/dev/null
pkill -f /usr/local/lib/ollama/llama-server 2>/dev/null

wait   # hosted runs (each has its own deadline)
$P -m bench.ingest --html $E/recipe_corpus/html --out $O/ingest.json > $LOG/ingest.log 2>&1
$P -m bench.leaderboard --runs $O | tee $LOG/leaderboard.txt
cat $LOG/ingest.log
echo "end $(date)"
