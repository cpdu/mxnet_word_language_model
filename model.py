import math
import mxnet as mx
from mxnet.gluon import nn, rnn

class RNNModel(nn.Block):
    """Container block with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, **kwargs):
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
            self.decoder = nn.Dense(ntoken, in_units=nhid, weight_initializer=mx.init.Uniform(0.1))

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.nhid)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)