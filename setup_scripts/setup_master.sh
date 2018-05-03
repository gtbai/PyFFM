#!/bin/zsh

# generate a new pair of ssh keys
ssh-keygen -t rsa

# set up ssh link to self
cat ~/.ssh/id_rsa.pub >> /users/gtbai/.ssh/authorized_keys

# set up ssh links from master to workers
for idx in {1..4}
do
    worker_name=cp-$idx
    echo "Creating SSH link to $worker_name..."
    cat ~/.ssh/id_rsa.pub | sudo ssh $worker_name "cat >> /users/gtbai/.ssh/authorized_keys; exit"
done

# copy repo to each worker
for idx in {1..4}
do
    worker_name=cp-$idx
    echo "Sending repo to $worker_name..."
    scp -r /mnt/project/DistributedFFM $worker_name:/mnt/project/
    ## ssh $worker_name "source setup_node.sh"
done
