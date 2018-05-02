import math
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn, rnn
import mxnet.autograd as ag

class RNNModel(nn.Block):
    """Container block with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, class_based=False, cls2wordnum=None, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.nhid = nhid

        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp, weight_initializer=mx.init.Uniform(0.1))
            stdv = 1.0 / math.sqrt(nhid)
            uniform_stdv = mx.init.Uniform(stdv)
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(rnn, rnn_type)(nhid, nlayers, input_size=ninp, dropout=dropout,
                                                  i2h_weight_initializer=uniform_stdv,
                                                  h2h_weight_initializer=uniform_stdv,
                                                  i2h_bias_initializer=uniform_stdv,
                                                  h2h_bias_initializer=uniform_stdv)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("Invalid model %s. Options are RNN_RELU, RNN_TANH, LSTM, and GRU." % rnn_type)
                self.rnn = rnn.RNN(nhid, nlayers, input_size=ninp, dropout=dropout, activation=nonlinearity,
                                   i2h_weight_initializer=uniform_stdv,
                                   h2h_weight_initializer=uniform_stdv,
                                   i2h_bias_initializer=uniform_stdv,
                                   h2h_bias_initializer=uniform_stdv)
            # self.decoder = nn.Dense(ntoken, in_units=nhid, weight_initializer=mx.init.Uniform(0.1))
            uniform_init = mx.init.Uniform(0.1)
            if class_based:
                self.class_based = True
                for class_id, word_num in enumerate(cls2wordnum):
                    setattr(self, 'word_decoder' + str(class_id), nn.Dense(word_num, in_units=nhid, weight_initializer=uniform_init))
                self.ncls = len(cls2wordnum)
                self.class_decoder = nn.Dense(self.ncls, in_units=nhid, weight_initializer=uniform_init)
            else:
                self.class_based = False
                self.decoder = nn.Dense(ntoken, in_units=nhid, weight_initializer=uniform_init)

    def forward(self, inputs, hidden, cls2idxlst=None):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output_flat = output.reshape((-1, self.nhid))
        if self.class_based:
            if ag.is_training():
                cls_decoded = self.class_decoder(output_flat).reshape((output.shape[0], output.shape[1], -1))
                word_decoded = []
                for class_id in range(self.ncls):
                    words = cls2idxlst[class_id]
                    if len(words) == 0:
                        word_decoded.append(None)
                        continue
                    words_in_class = nd.take(output_flat, cls2idxlst[class_id])
                    word_decoded.append(getattr(self, 'word_decoder' + str(class_id))(words_in_class))
                return cls_decoded, word_decoded, hidden
            else:
                word_decoded = []
                cls_decoded = self.class_decoder(output_flat).reshape((output.shape[0], output.shape[1], -1))
                for class_id in range(self.ncls):
                    word_decoded.append(getattr(self, 'word_decoder' + str(class_id))(output_flat)
                                        .reshape((output.shape[0], output.shape[1], -1)))
                return cls_decoded, word_decoded, hidden
        else:
            decoded = self.decoder(output_flat)
            return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)