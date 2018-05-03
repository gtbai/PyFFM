# generate a new pair of ssh keys
ssh-keygen -t rsa

# set up ssh links from to master node
cat ~/.ssh/id_rsa.pub | sudo ssh ctl "cat >> /users/gtbai/.ssh/authorized_keys"

# set up ssh links to worker nodes
for idx in {1..4}
do
    worker_name=cp-$idx
    echo "Creating SSH link to $worker_name..."
    cat ~/.ssh/id_rsa.pub | sudo ssh $worker_name "cat >>  /users/gtbai/.ssh/authorized_keys; exit"
done
