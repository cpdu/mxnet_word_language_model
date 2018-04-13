import math
import time
import argparse
import mxnet as mx
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
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--num-gpu', type=int, default=0,
                    help='number of GPUs')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--kv-store', type=str,  default='local',
                    help='kvstore type')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
mx.random.seed(args.seed)

# try to use GPU
if args.num_gpu:
    try:
        context = [mx.gpu(i) for i in range(args.num_gpu)]
        for c in context:
            _ = nd.array([0], ctx=c)
    except:
        raise RuntimeError("You don't have {:d} available GPUs.".format(args.num_gpu))
else:
    context = [mx.cpu()]

# load data
corpus = data.Corpus(args.data)
vocab_size = len(corpus.dictionary)

def batchify(data, batch_size):
    """bachify data shape to (num_batches, batch_size)"""
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

# split data according to context
train_data = gluon.utils.split_and_load(batchify(corpus.train, args.batch_size), context, batch_axis=1)
val_data = gluon.utils.split_and_load(batchify(corpus.valid, args.batch_size), context, batch_axis=1)
test_data = gluon.utils.split_and_load(batchify(corpus.test, args.batch_size), context, batch_axis=1)

# create a model and initialize
model = model.RNNModel(args.model, vocab_size, args.emsize, args.nhid, args.nlayers, args.dropout)
model.initialize(ctx=context)
trainer = gluon.Trainer(model.collect_params(),
                        optimizer='sgd',
                        optimizer_params={'learning_rate': args.lr},
                        kvstore=args.kv_store)
criterion = gluon.loss.SoftmaxCrossEntropyLoss()

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() if not isinstance(i, list) else detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def get_batch(source, i):
    seq_len = min(args.bptt, source[0].shape[0] - 1 - i)
    data = [x[i : i + seq_len] for x in source]
    target = [x[i + 1 : i + 1 + seq_len].reshape((-1,)) for x in source]
    return data, target

def evaluate(data_source):
    total_loss = 0.0
    ntotal = 0
    hidden = [model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size // len(context), ctx=ctx) for ctx in context]
    for i in range(0, data_source[0].shape[0] - 1, args.bptt):
        data, target = get_batch(data_source, i)
        losses = [criterion(model(d, h)[0], t) for d, h, t in zip(data, hidden, target)]
        loss = 0
        for l in losses:
            loss += mx.nd.sum(l).asscalar()
        total_loss += loss
        ntotal += losses[0].size * len(context)
    return total_loss / ntotal

def train():
    total_loss = 0.0
    start_time = time.time()
    hidden = [model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size // len(context), ctx=ctx) for ctx in context]
    for batch, i in enumerate(range(0, train_data[0].shape[0] - 1, args.bptt)):
        data, target = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = detach(hidden)
        with autograd.record():
            losses = [criterion(model(d, h)[0], t) for d, h, t in zip(data, hidden, target)]
        for l in losses:
            l.backward()

        # sum over losses on each context
        loss = 0
        for l in losses:
            loss += mx.nd.sum(l).asscalar()
        total_loss += loss / args.bptt / args.batch_size

        # grad clipping helps prevent the exploding gradient problem in RNNs / LSTMs.
        for c in context:
            grads = [p.grad(c) for p in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, args.clip * args.bptt * args.batch_size / len(context))

        # automatically average gradients in trainer.step
        trainer.step(args.bptt * args.batch_size)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} '
                  '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch + 1, batch, len(train_data[0]) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.0
            start_time = time.time()

lr = args.lr
best_val_loss = None
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train()
    training_time = time.time() - epoch_start_time

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

# Load the best saved model.
model.load_params(args.save, context)
# Run on test data.
test_loss = evaluate(test_data)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
