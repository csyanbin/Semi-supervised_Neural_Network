#!/bin/bash
for k in $( seq 1 2 )
do
    for res in 10 16 20 32 64
    do
        n=20
        python3 coil_data.py --seed=${k} --num_labelled=${n} --resolution=${res}
    done
done
