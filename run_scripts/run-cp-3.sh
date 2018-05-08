#!/bin/zsh

for batch_size in 8792 16384 32768 65536
do
    for learning_rate in 0.001 0.01 0.1
    do
        echo "now running mat-FFM.py with batch size $batch_size and learning rate $learning_rate..."
        python mat-FFM.py --batch_size=$batch_size --learning_rate=$learning_rate &> output/big-dataset/output-${batch_size}-${learning_rate}.txt
    done
done

