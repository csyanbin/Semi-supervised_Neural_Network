#!/bin/bash
# 1:label 2:ratio 3:dataset 4:resolution 5:train_nums (default20)
echo $1 $2 $3 $4 $5
for k in 1  
do
    mkdir -p $3/log_$2
    python3 ladder_time.py $1 ${k} $2 $3 $4 $5 2>&1 | tee $3/log$4_$2/log_label$1_${k}_$2_$5.time.log
    echo $1 ${k} $2 $3 $4 $5
done
