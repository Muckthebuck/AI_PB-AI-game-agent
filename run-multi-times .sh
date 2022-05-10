#!/bin/bash
run_times =2

echo "Going to run model for ${run_times} time(s)"
for k in $(seq 1 ${run_times})
do
  python -m referee 7 emotionaldamage disappointment -l game${k}
done