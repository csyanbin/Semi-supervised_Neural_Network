#!/bin/bash
echo $1
for k in $( seq 1 20 )
do
    python3 ladder.py $1 ${k} 2>&1 | tee log/log_label$1_${k}.log
    echo $1 ${k}
done
