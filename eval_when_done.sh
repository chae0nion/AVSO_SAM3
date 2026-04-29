#!/bin/bash
# 각 ablation 실험이 끝나면 자동으로 lvos-evaluation 돌리는 스크립트

LVOS_EVAL=/workspace/lvos-evaluation/evaluation_method.py
LVOS_PATH=/workspace/sam3-main/LVOS2
OUT=/workspace/sam3-main/output

wait_pid() {
    local pid=$1
    while kill -0 $pid 2>/dev/null; do
        sleep 30
    done
}

run_eval() {
    local name=$1
    local pred=$2
    local log=$3
    echo "[$(date '+%H:%M:%S')] Starting eval for $name ..."
    python $LVOS_EVAL \
        --lvos_path $LVOS_PATH \
        --results_path $pred \
        --set valid \
        --task semi-supervised \
        --mp_nums 8 \
        > $log 2>&1
    echo "[$(date '+%H:%M:%S')] Eval done for $name → $log"
}

# partb: PID 21700
echo "[$(date '+%H:%M:%S')] Waiting for partb (PID 21700)..."
wait_pid 21700
run_eval "partb" "$OUT/ablation_partb" "$OUT/eval_partb.log"

# partc: PID 131582
echo "[$(date '+%H:%M:%S')] Waiting for partc (PID 131582)..."
wait_pid 131582
run_eval "partc" "$OUT/ablation_partc" "$OUT/eval_partc.log"

# abc: PID 131678
echo "[$(date '+%H:%M:%S')] Waiting for abc (PID 131678)..."
wait_pid 131678
run_eval "abc" "$OUT/ablation_abc" "$OUT/eval_abc.log"

echo "[$(date '+%H:%M:%S')] All evaluations done."
