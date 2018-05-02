import math
import time
import argparse
from collections import defaultdict
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
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cls', action='store_true',
                    help='use class-based training')
parser.add_argument('--ncls', type=int, default=20,
                    help='number of classes')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
mx.random.seed(args.seed)

# try to use GPU
if args.cuda:
    try:
        context = mx.gpu()
        _ = nd.array([0], ctx=context)
    except:
        raise RuntimeError("You have no cuda devices while using --cuda.")
else:
    context = mx.cpu()


# load data
corpus = data.Corpus(args.data, class_num=args.cls*args.ncls)
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

model = model.RNNModel(args.model, vocab_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.cls, corpus.dictionary.cls2wordnum)
model.initialize(ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args.lr, 'momentum': 0, 'wd': 0})
criterion = gluon.loss.SoftmaxCrossEntropyLoss()

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

def classify(classes, cindices):
    idxdict = defaultdict(lambda: [])
    cidxdict = defaultdict(lambda: [])
    for i, (cls, cidx) in enumerate(zip(classes, cindices)):
        idxdict[cls].append(i)
        cidxdict[cls].append(cidx)
    cls2idxlst = []
    cls2cidxlst = []
    for cls in range(args.ncls):
        cls2idxlst.append(nd.array(idxdict[cls], ctx=context))
        cls2cidxlst.append(nd.array(cidxdict[cls], ctx=context))
    return cls2idxlst, cls2cidxlst

if args.cls:
    idx2class = nd.array(corpus.dictionary.idx2class, ctx=context, dtype='int32')
    idx2cidx = nd.array(corpus.dictionary.idx2cidx, ctx=context, dtype='int64')

def evaluate(data_source, class_based=False):
    total_loss = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args.batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args.bptt):
        data, target = get_batch(data_source, i)
        if class_based:
            tar_classes = nd.take(idx2class, target)
            tar_cindices = nd.take(idx2cidx, target)
            cls2idxlst, cls2cidxlst = classify(tar_classes.asnumpy(), tar_cindices.asnumpy())
            cls_output, word_output, hidden = model(data, hidden)
            total_word_num = cls_output.shape[0] * cls_output.shape[1]
            word_loss = None
            for cls in range(args.ncls):
                if len(cls2idxlst[cls]) == 0:
                    continue
                selected_output = nd.take(word_output[cls].reshape((total_word_num, -1)), cls2idxlst[cls])
                cindices = cls2cidxlst[cls]
                if word_loss is None:
                    word_loss = criterion(selected_output, cindices)
                else:
                    word_loss = nd.concat(word_loss, criterion(selected_output, cindices), dim=0)
            cls_loss = criterion(cls_output.reshape((-1, args.ncls)), tar_classes)
            total_loss += nd.sum(word_loss).asscalar() + nd.sum(cls_loss).asscalar()
            ntotal += cls_loss.size
        else:
            output, hidden = model(data, hidden)
            loss = criterion(output, target)
            total_loss += mx.nd.sum(loss).asscalar()
            ntotal += loss.size
    return total_loss / ntotal

def train(class_based=False):
    total_loss = 0.0
    start_time = time.time()
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)

    for batch, i in enumerate(range(0, train_data.shape[0] - 1, args.bptt)):
        data, target = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = detach(hidden)
        if class_based:
            tar_classes = nd.take(idx2class, target)
            tar_cindices = nd.take(idx2cidx, target)
            cls2idxlst, cls2cidxlst = classify(tar_classes.asnumpy(), tar_cindices.asnumpy())
            with autograd.record():
                cls_output, word_output, hidden = model(data, hidden, cls2idxlst)

                word_loss = None
                for cls in range(args.ncls):
                    cindices = cls2cidxlst[cls]
                    if len(cindices) == 0:
                        continue
                    if word_loss is not None:
                        word_loss = nd.concat(word_loss, criterion(word_output[cls], cindices), dim=0)
                    else:
                        word_loss = criterion(word_output[cls], cindices)
                cls_loss = criterion(cls_output.reshape((-1, args.ncls)), tar_classes)
                loss = word_loss + cls_loss
        else:
            with autograd.record():
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
        loss.backward()

        grads = [p.grad(context) for p in model.collect_params().values()]
        # grad clipping helps prevent the exploding gradient problem in RNNs / LSTMs.
        gluon.utils.clip_global_norm(grads, args.clip * args.bptt * args.batch_size)
        trainer.step(args.bptt * args.batch_size)
        total_loss += mx.nd.sum(loss).asscalar() / loss.shape[0]

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} '
                  '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch + 1, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.0
            start_time = time.time()

lr = args.lr
best_val_loss = None
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(args.cls)
    training_time = time.time() - epoch_start_time

    val_loss = evaluate(val_data, args.cls)
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
test_loss = evaluate(test_data, args.cls)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
