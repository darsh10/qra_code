
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

class DAN(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(DAN, DAN).add_config(cfgparser)
        cfgparser.add_argument("--n_d", "--d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_layers", "--depth", type=int, help="number of non-linear layers")


    def __init__(self, embedding_layer, configs):
        super(DAN, self).__init__(configs)
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.num_layers = configs.num_layers or 1
        self.activation = configs.activation or 'tanh'
        self.n_e = embedding_layer.n_d
        self.n_d = configs.n_d or 300
        self.dropout = configs.dropout or 0.0
        self.use_cuda = configs.cuda

        self.dropout_op = nn.Dropout(self.dropout)
        activation_module = get_activation_module(self.activation)
        self.seq = seq = nn.Sequential()
        for i in range(self.num_layers):
            n_in = self.n_d if i>0 else self.n_e
            n_out = self.n_d
            seq.add_module('linear-{}'.format(i),
                nn.Linear(n_in, n_out)
            )
            seq.add_module('activation-{}'.format(i),
                activation_module()
            )
            if self.dropout > 0:
                seq.add_module('dropout-{}'.format(i),
                    nn.Dropout(p=configs.dropout)
                )

        self.n_out = self.n_d if self.num_layers > 0 else self.n_e
        self.build_output_op()

    def forward(self, batch):
        # batch is of size (len, batch_size)
        emb = self.embedding(batch)  # (len, batch_size, n_e)
        emb = Variable(emb.data)
        #emb = emb.detach()
        assert emb.dim() == 3

        if self.dropout > 0:
            emb = self.dropout_op(emb)

        # get mask
        padid = self.embedding_layer.padid
        mask = (batch != padid).type(torch.FloatTensor)
        if self.use_cuda:
            mask = mask.cuda()
        colsum = torch.sum(mask, 0).view(-1) # (batch_size,)
        mask = mask[:,:,None].expand_as(emb)  # (len, batch_size, n_e)

        # average word embedding
        sum_emb = torch.sum(emb*mask, 0).view(emb.size(1), -1) # (batch_size, n_e)
        avg_emb = sum_emb / colsum[:,None].expand_as(sum_emb) # (batch_size, n_e)

        # pass through non-linear layers
        out = self.seq(avg_emb) if self.num_layers > 0 else avg_emb

        return out


