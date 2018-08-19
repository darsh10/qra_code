
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .basic import get_activation_module
from .basic import indent

class DAN_DOMAIN(nn.Module):

    @staticmethod
    def add_config(cfgparser):
        cfgparser.add_argument("--n_d", "--d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_layers", "--depth", type=int, help="number of non-linear layers")


    def __init__(self, embedding_layer, configs):
        super(DAN_DOMAIN, self).__init__()
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.num_layers = configs.num_layers or 1
        self.activation = configs.activation or 'tanh'
        self.n_e = embedding_layer.n_d
        self.n_d = configs.n_d or 300
        self.dropout = configs.dropout or 0.0
        self.use_cuda = configs.cuda

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

        n_out = self.n_d if self.num_layers > 0 else self.n_e
        self.output_layer = nn.Linear(2*n_out, 2)

    def forward(self, batch_pair):
        # pair_left or pair_right is of size (len, batch_size)
        pair_left, pair_right = batch_pair

        # size (batch, n_d)
        out_left = self.forward_one_side(pair_left)
        out_right = self.forward_one_side(pair_right)
        out = torch.cat(( out_left, out_right ), 1 )#out_left * out_right

        out = self.output_layer.forward(out)
        return out

    def forward_one_side(self, batch):
        # batch is of size (len, batch_size)
        emb = self.embedding(batch)  # (len, batch_size, n_e)
        emb = Variable(emb.data)
        #emb = emb.detach()
        assert emb.dim() == 3

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

    def __repr__(self):
        text = "DAN (\n{}\n{}\n{}\n)".format(
            indent(str(self.embedding), 2),
            indent(str(self.seq), 2),
            indent(str(self.output_layer), 2)
        )
        return text


