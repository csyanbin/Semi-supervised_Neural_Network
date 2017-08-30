#!/bin/bash
for k in $( seq 1 20 )
do
    for l in $( seq 1 6)
    do
        n=$((l*7))
        python3 pollen_data.py --seed=${k} --num_labelled=${n}
    done
done
