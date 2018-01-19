#!/bin/bash
 # compute time for all label and unlabeled data vari from 5% to 50%
for k in $( seq 1 10 )
do
    for tn in 4 9 18 27 36 45
    do
        n=7
        python3 pollen_data.py --seed=${k} --num_labelled=${n} --train_nums=${tn}
    done
done
