#!/bin/bash
cd /Users/moritzbecker/projekt/as-pilot
source ../venv/bin/activate 2>/dev/null
export PYTHONHASHSEED=0 PYTHONPATH=.
for mode in off inherit; do
  for ceil in 1.0 2.0; do
    for seed in 1001 1002 1003; do
      out="sweep_decisive/g24x18_${mode}_c${ceil}"
      python scripts/m1_pilot.py --exp learn --seed $seed --ticks 5000 \
        --grid-w 24 --grid-h 18 --pop 30 --snapshot-interval 250 \
        --physics v2 --taxis --age-structured-founders \
        --plant-ceiling-scale $ceil --respawn-mode $mode \
        --out "$out" >> sweep_decisive/progress.log 2>&1
      echo "done $mode c$ceil seed$seed" >> sweep_decisive/progress.log
    done
  done
done
echo "SWEEP_COMPLETE" >> sweep_decisive/progress.log
