#!/bin/bash
# CLASSIC E1a queue: two lanes x 4 threads (= 8 threads), CPU only, nice 15; marker CLASSIC/E1A_DONE
D=<RUN_ROOT>/study/CLASSIC
PY=python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
lane() { for ds in "$@"; do nice -n 15 $PY -s -B $D/code/classic_ds.py $ds --threads 4 >> $D/logs/e1a_$ds.log 2>&1; echo "$ds exit $?" >> $D/logs/e1a_lanes.log; done; }
lane office baby pantry scientific instruments &
lane clothing arts sports microlens &
wait
echo "DONE $(date)" > $D/E1A_DONE
