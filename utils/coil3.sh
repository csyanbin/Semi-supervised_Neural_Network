#!/bin/bash
for k in $( seq 1 2 )
do
    # for tn in 4 7 14 22 29 32
    for tn in 3 4 7 14 15 21 22 29 36 
    do
        n=20
        python3 coil_data.py --seed=${k} --num_labelled=${n} --train_nums=${tn}
    done
done
