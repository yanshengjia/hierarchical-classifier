#!/bin/bash
cd ../src

declare -a algo=('gbdt' 'svc' 'rf' 'lr')

for c1 in "${algo[@]}"
do
    for c2 in "${algo[@]}"
    do
        for c3 in "${algo[@]}"
        do
            for c4 in "${algo[@]}"
            do
                for c5 in "${algo[@]}"
                do
                    python run.py --train --c1 $c1 --c2 $c2 --c3 $c3 --c4 $c4 --c5 $c5
                done
            done
        done
    done
done

echo "Done with grid search!"