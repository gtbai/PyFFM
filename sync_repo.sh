#!/bin/zsh
# copy repo to each worker
cd ~
for idx in {1..4}
do
    worker_name=cp-$idx
    echo "Sending repo to $worker_name..."
    ssh $worker_name "rm -rf ~/DistributedFFM"
    scp -r ~/DistributedFFM $worker_name:~/
    ## ssh $worker_name "source setup_node.sh"
done
