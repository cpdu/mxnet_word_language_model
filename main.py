import os
import mxnet as mx

import math
import time
import argparse
import mxnet.ndarray as nd
from mxnet import gluon, autograd

import data
import model

parser = argparse.ArgumentParser(description='Mxnet PTB RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--deviceid', type=int, default=0,
                    help='device(GPU) id')
parser.add_argument('--navg', type=int, default=20,
                    help='model average interval')
parser.add_argument('--ita', type=float, default=0.1,
                    help='block momentum coefficient')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--ip', type=str, default='127.0.0.1',
                    help='scheduler IP')
parser.add_argument('--port', type=str, default='9000',
                    help='scheduler port')
parser.add_argument('--num-server', type=str, default='1',
                    help='number of servers')
parser.add_argument('--num-worker', type=str, default='1',
                    help='number of workers')
parser.add_argument('--verbose', type=str, default='0',
                    help='log verbose')
parser.add_argument('--kv-store', type=str,  default='dist_sync',
                    help='kvstore type')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
mx.random.seed(args.seed)

# try to use GPU
deviceid = args.deviceid
try:
    context = mx.gpu(deviceid)
    _ = nd.array([0], ctx=context)
except:
    raise RuntimeError("You have only {:d} GPU(s).".format(deviceid))

# configure distributed training env
os.environ.update({"DMLC_ROLE": "worker",
                   "DMLC_PS_ROOT_URI": args.ip,
                   "DMLC_PS_ROOT_PORT": args.port,
                   "DMLC_NUM_SERVER": args.num_server,
                   "DMLC_NUM_WORKER": args.num_worker,
                   "PS_VERBOSE": args.verbose})

# create kvstore and set its optimizer
kv = mx.kv.create(args.kv_store)
if kv.rank == 0:
    optim = mx.optimizer.SGD(learning_rate=-1/kv.num_workers, wd=-kv.num_workers)
    kv.set_optimizer(optim)

# load data
corpus = data.Corpus(args.data)
vocab_size = len(corpus.dictionary)

def batchify(data, batch_size):
    """bachify data shape to (num_batches, batch_size)"""
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

train_data = batchify(corpus.train, args.batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args.batch_size).as_in_context(context)
test_data = batchify(corpus.test, args.batch_size).as_in_context(context)

model = model.RNNModel(args.model, vocab_size, args.emsize, args.nhid, args.nlayers, args.dropout)
model.initialize(ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args.lr, 'momentum': 0, 'wd': 0})
criterion = gluon.loss.SoftmaxCrossEntropyLoss()

# init kvstore values
for i, p in enumerate(model.collect_params().values()):
    kv.init(i, p.data())
lr_key = i+1
kv.init(lr_key, nd.zeros((1)))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def get_batch(source, i):
    seq_len = min(args.bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def evaluate(data_source):
    total_loss = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args.batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args.bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        loss = criterion(output, target)
        total_loss += mx.nd.sum(loss).asscalar()
        ntotal += loss.size
    return total_loss / ntotal

# ensure all models are identical at the beginning
for j, p in enumerate(model.collect_params().values()):
    kv.push(j, p.data(), priority=-j)
for j, p in enumerate(model.collect_params().values()):
    kv.pull(j, p.data(), priority=-j)

# about block momentum
p_old = []
p_new = []
delta_t = []
for p in model.collect_params().values():
    p_old.append(p.data().copy())
    p_new.append(nd.ones(p.shape, ctx=context))
    delta_t.append(nd.zeros(p.shape, ctx=context))

def train(offset):
    total_loss = 0.0
    start_time = time.time()
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
    param_num = len(model.collect_params().values())
    for batch, i in enumerate(range(0, train_data.shape[0] // kv.num_workers - 1, args.bptt)):
        data, target = get_batch(train_data, i+offset)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = detach(hidden)

        with autograd.record():
            output, hidden = model(data, hidden)
            loss = criterion(output, target)
        loss.backward()
        total_loss += mx.nd.sum(loss).asscalar() / loss.shape[0]

        # grad clipping helps prevent the exploding gradient problem in RNNs / LSTMs.
        rescale = args.bptt * args.batch_size
        grads = [p.grad(context) for p in model.collect_params().values()]
        gluon.utils.clip_global_norm(grads, args.clip * rescale)

        # update local model
        trainer.step(rescale)

        # avarage parameters from each workers
        if kv.num_workers > 1 and batch % args.navg == 0 and batch != 0:
            for j, p in enumerate(model.collect_params().values()):
                kv.push(j, p.data(), priority=-j)
            for j, p in enumerate(model.collect_params().values()):
                kv.pull(j, p_new[j], priority=-j)
                delta_t[j][:] = args.ita * delta_t[j] + (p_new[j] - p_old[j])
                p.set_data(p_old[j] + delta_t[j])
                p_old[j][:] = p_new[j]

        if kv.rank == 0 and batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} '
                  '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch + 1, batch, len(train_data) // args.bptt // kv.num_workers, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.0
            start_time = time.time()

lr = args.lr
best_val_loss = None
data_part = kv.rank
data_len = len(train_data)
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(data_part * data_len // kv.num_workers)
    training_time = time.time() - epoch_start_time
    data_part = (data_part + 1) % kv.num_workers

    if kv.rank == 0:
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch + 1, training_time, val_loss, math.exp(val_loss)))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            model.save_params(args.save)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            trainer.set_learning_rate(lr)
        kv.push(lr_key, nd.array([lr]))
    else:
        kv.push(lr_key, nd.array([0]))
        lr_arr = nd.zeros((1))
        kv.pull(lr_key, lr_arr)
        lr = lr_arr.asscalar() * kv.num_workers
        trainer.set_learning_rate(lr)

if kv.rank == 0:
    # Load the best saved model.
    model.load_params(args.save, context)
    # Run on test data.
    test_loss = evaluate(test_data)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)

