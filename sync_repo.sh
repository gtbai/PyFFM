#!/bin/zsh
# copy repo to each worker
cd ~
for idx in {2..4}
do
    worker_name=cp-$idx
    echo "Sending repo to $worker_name..."
    ssh $worker_name "rm -rf /mnt/project/DistributedFFM"
    scp -r /mnt/project/DistributedFFM $worker_name:/mnt/project
    ## ssh $worker_name "source setup_node.sh"
done
