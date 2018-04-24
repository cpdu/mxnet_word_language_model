#!/bin/bash

# parallelization parameters
kvstore=device
num_gpu=2

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

python3 -u main.py --data ${data} --model ${model} --emsize ${emsize} \
                   --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} \
                   --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} \
                   --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --kv-store ${kvstore} --num-gpu ${num_gpu}
