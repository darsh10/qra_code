import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

class DOM_LSTM(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(DOM_LSTM, DOM_LSTM).add_config(cfgparser)
        cfgparser.add_argument("--n_d", "--d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_lstm", type=int, help="number of stacking lstm layers")
        cfgparser.add_argument("--num_layers", "--depth", type=int, help="number of non-linear layers")
        cfgparser.add_argument("--bidirectional", "--bidir", action="store_true", help="use bi-directional LSTM")

    def __init__(self, embedding_layer, configs):
        super(DOM_LSTM, self).__init__(configs)
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.n_e = embedding_layer.n_d
        self.n_d = configs.n_d or 300
        self.activation = configs.activation or 'tanh'
        self.dropout = configs.dropout or 0.0
        self.num_lstm = configs.num_lstm or 1
        self.num_layers = configs.num_layers or 0
        self.bidirectional = configs.bidirectional
        self.use_cuda = configs.cuda

        self.dropout_op = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            input_size = self.n_e,
            hidden_size = self.n_d,
            num_layers = self.num_lstm,
            dropout = self.dropout,
            bidirectional = self.bidirectional
        )

        activation_module = get_activation_module(self.activation)
        self.seq = seq = nn.Sequential()
        actual_d = self.n_d*2 if self.bidirectional else self.n_d
        for i in range(self.num_layers):
            seq.add_module('linear-{}'.format(i),
                nn.Linear(actual_d, actual_d)
            )
            seq.add_module('activation-{}'.format(i),
                activation_module()
            )
            if self.dropout > 0:
                seq.add_module('dropout-{}'.format(i),
                    nn.Dropout(p=configs.dropout)
                )
        self.n_out = actual_d

        self.domain_classifier = False

        self.domain_classifier_layer = nn.Sequential()
        self.domain_classifier_layer.add_module('linear-{}'.format(0),
            nn.Linear(self.n_out, self.n_out)
        )
        self.domain_classifier_layer.add_module('activation-{}'.format(0),
            nn.ReLU()
        )
        self.domain_classifier_layer.add_module('linear-{}'.format(1),
            nn.Linear(self.n_out, self.n_out)
        )
        self.domain_classifier_layer.add_module('activation-{}'.format(1),
            nn.ReLU()
        )
        self.domain_classifier_layer.add_module('linear-{}'.format(2),
            nn.Linear(self.n_out, 2)
        )

        self.build_output_op()

    def forward(self, batch_pair):

       if not self.domain_classifier:
            # pair_left or pair_right is of size (len, batch_size)
            pair_left, pair_right = batch_pair

            # size (batch, n_d)
            out_left = self.forward_one_side(pair_left)
            out_right = self.forward_one_side(pair_right)

            return self.compute_output(out_left, out_right)

       else:
            out = self.forward_one_side(batch_pair)
            out = self.domain_classifier_layer.forward(out)

            return out

    def forward_one_side(self, batch):
        # batch is of size (len, batch_size)
        emb = self.embedding(batch)  # (len, batch_size, n_e)
        emb = Variable(emb.data)
        assert emb.dim() == 3

        if self.dropout > 0:
            emb = self.dropout_op(emb)

        output, hidden = self.lstm(emb)

        if self.dropout > 0:
            output = self.dropout_op(output)

        #output = output[-1] # last state (batch_size, d)

        # get mask
        padid = self.embedding_layer.padid
        mask = (batch != padid).type(torch.FloatTensor)
        if self.use_cuda:
            mask = mask.cuda()
        colsum = torch.sum(mask, 0).view(-1) # (batch_size,)
        mask = mask[:,:,None].expand_as(output)  # (len, batch_size, d)

        # average over representations
        sum_emb = torch.sum(output*mask, 0).view(output.size(1), -1) # (batch_size, d)
        avg_emb = sum_emb / colsum[:,None].expand_as(sum_emb) # (batch_size, d)
        output = avg_emb

        # pass through non-linear layers
        if self.num_layers > 0:
            output = self.seq(output)

        return output


