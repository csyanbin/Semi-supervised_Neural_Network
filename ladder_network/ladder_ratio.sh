#!/bin/bash
echo $1 $2 $3
for k in $( seq 1 20 )
do
    mkdir -p $3/log_$2
    python3 ladder.py $1 ${k} $2 $3 2>&1 | tee $3/log_$2/log_label$1_${k}_$2.log
    echo $1 ${k} $2 $3
done
