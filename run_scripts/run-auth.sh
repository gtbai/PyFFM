#!/bin/zsh

for batch_size in 8192 4096 2048 1024 512 256 128
do
    echo "now running mat-FFM-auth.py with batch size $batch_size..."
    python mat-FFM-auth.py --batch_size=$batch_size &> output-auth/output-auth-$batch_size.txt
done

