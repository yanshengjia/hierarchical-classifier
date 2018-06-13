#!/bin/bash
cd ../src

declare -a algo=('gbdt' 'svc' 'rf' 'lr')

for algo1 in "${algo[@]}"
do
    for algo2 in "${algo[@]}"
    do
        for algo3 in "${algo[@]}"
        do
            for algo4 in "${algo[@]}"
            do
                for algo5 in "${algo[@]}"
                do
                    python run.py --train --algo1_1 $algo1 --algo1_2 $algo2 --algo1_3 $algo3 --algo1_4 $algo4 --algo1_5 $algo5
                done
            done
        done
    done
done

echo "Done with grid search!"