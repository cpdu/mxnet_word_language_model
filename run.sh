#!/bin/bash

# parallelization parameters
num_workers=2
rank=0
ip=127.0.0.1
port=9000
verbose=0
npush=8
npull=24
kvstore=dist_async

# training parameters
epoch=10
batch_size=20
bptt=35
nhid=400
data=$HOME/dataset/wikitext-2
model=LSTM
emsize=200
nlayers=2
lr=20
clip=0.25
dropout=0.2
seed=1111
log_interval=2
save=model.pt

nvidia-smi
hostname
echo $CUDA_VISIBLE_DEVICES

# get number of GPUs
devices=$CUDA_VISIBLE_DEVICES
OLD_IFS="$IFS"
IFS=","
device_list=($devices)
IFS="$OLD_IFS"
gpu_num=${#device_list[@]}

# start the scheduler
if [ "$rank" == "0" ]; then
    python3 -u server.py --role scheduler --ip ${ip} --port ${port} --num-server ${num_workers} \
                         --num-worker ${num_workers} --verbose ${verbose} &
fi                     

# start servers and workers
i=0
while [ "${rank}" != "$((num_workers-1))" -a "${i}" != "$((gpu_num-1))" ]
do
    python3 -u server.py --role server --ip ${ip} --port ${port} --num-server ${num_workers} \
                         --num-worker ${num_workers} --verbose ${verbose} &
    python3 -u main.py --deviceid ${i} --data ${data} --model ${model} --emsize ${emsize} \
                       --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} \
                       --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} \
                       --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --npush ${npush} \
                       --npull ${npull} --kv-store ${kvstore} --ip ${ip} --port ${port} \
                       --num-server ${num_workers} --num-worker ${num_workers} --verbose ${verbose} &
    rank=$(($rank+1))
    i=$(($i+1))
done

python3 -u server.py --role server --ip ${ip} --port ${port} --num-server ${num_workers} \
                     --num-worker ${num_workers} --verbose ${verbose} &
python3 -u main.py --deviceid ${i} --data ${data} --model ${model} --emsize ${emsize} \
                   --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} \
                   --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} \
                   --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --npush ${npush} \
                   --npull ${npull} --kv-store ${kvstore} --ip ${ip} --port ${port} \
                   --num-server ${num_workers} --num-worker ${num_workers} --verbose ${verbose}
