#!/bin/bash
G=<RUN_ROOT>/study/MESF
PY=python
for ds in baby pantry scientific sports; do
  nice -n 15 $PY -u $G/code/mesfres_sc.py $ds $G/sc_cache --threads 8 >> $G/logs/sc_$ds.log 2>&1
done
echo done > $G/markers/SC_DONE
