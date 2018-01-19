#!/bin/bash
# compute for resolution varies from 10 to 32
for k in $( seq 1 20 )
do
    for res in 10 16 20 32
    do
        n=7
        python3 pollen_data.py --seed=${k} --num_labelled=${n} --resolution=${res}
    done
done
