
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

class DOMDAN(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(DOMDAN, DOMDAN).add_config(cfgparser)
        cfgparser.add_argument("--n_d", "--d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_layers", "--depth", type=int, help="number of non-linear layers")


    def __init__(self, embedding_layer, configs):
        super(DOMDAN, self).__init__(configs)
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.num_layers = configs.num_layers or 1
        self.activation = configs.activation or 'tanh'
        self.n_e = embedding_layer.n_d
        self.n_d = configs.n_d or 300
        self.dropout = configs.dropout or 0.0
        self.use_cuda = configs.cuda
        self.domain_classifier = False
        #self.w = nn.Parameter(torch.Tensor([ self.n_e ]).cuda(), requires_grad=True)

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
        self.output_layer = nn.Linear(self.n_out, 2)
        #self.domain_classifier_layer = nn.Linear(n_out, 2)
        self.domain_classifier_layer = nn.Sequential()
        self.domain_classifier_layer.add_module('activation-{}'.format(0),
            activation_module()
        )
        self.domain_classifier_layer.add_module('linear-{}'.format(0),
            nn.Linear(self.n_out, 2)
        )
        self.build_output_op()

    def forward(self, batch_pair, ):

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
        #emb = emb.detach()
        assert emb.dim() == 3

        # get mask
        padid = self.embedding_layer.padid
        mask = (batch != padid).type(torch.FloatTensor)
        if self.use_cuda:
            mask = mask.cuda()
        colsum = torch.sum(mask, 0).view(-1) # (batch_size,)
        mask = mask[:,:,None].expand_as(emb)  # (len, batch_size, n_e)
        #mask = mask * self.w.expand_as(mask)
        #mask = F.tanh(mask)  

        # average word embedding
        sum_emb = torch.sum(emb*mask, 0).view(emb.size(1), -1) # (batch_size, n_e)
        avg_emb = sum_emb / colsum[:,None].expand_as(sum_emb) # (batch_size, n_e)

        # pass through non-linear layers
        out = self.seq(avg_emb) if self.num_layers > 0 else avg_emb

        return out

    def __repr__(self):
        text = "DOMDAN (\n{}\n{}\n{}\n)".format(
            indent(str(self.embedding), 2),
            indent(str(self.seq), 2),
            indent(str(self.output_layer), 2)
        )
        return text

