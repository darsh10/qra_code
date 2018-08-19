
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .basic import get_activation_module
from .basic import indent

class DOMSLC(nn.Module):

    @staticmethod
    def add_config(cfgparser):
        cfgparser.add_argument("--n_d", "--d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_layers", "--depth", type=int, help="number of non-linear layers")


    def __init__(self, embedding_layer, configs):
        super(DOMSLC, self).__init__()
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.num_layers = configs.num_layers or 1
        self.activation = configs.activation or 'tanh'
        self.n_e = embedding_layer.n_d
        self.n_d = configs.n_d or 300
        self.dropout = configs.dropout or 0.0
        self.use_cuda = configs.cuda
        self.batch_size = configs.batch_size
        self.domain_classifier = False

        n_out = self.n_d if self.num_layers > 0 else self.n_e
        self.output_layer = nn.Linear(n_out, 2)
        self.seq = self.lstm = nn.LSTM(input_size=self.n_e, hidden_size=self.n_d, dropout = self.dropout, batch_first=True).cuda()
        self.hidden = self.init_hidden(self.batch_size*2)
        self.domain_classifier_layer = nn.Linear(2*n_out, 2)

    def init_hidden(self, batch_size):
        
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.n_d).cuda())
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.n_d).cuda())
        return ((h0, c0))

    def forward(self, batch_pair):
        # pair_left or pair_right is of size (len, batch_size)
        pair_left, pair_right = batch_pair

        # size (batch, n_d)
        out_left = self.forward_one_side(pair_left)
        out_right = self.forward_one_side(pair_right)
        if not self.domain_classifier:
            out = out_left * out_right
            out = self.output_layer.forward(out)

        else:
            out = torch.cat(( out_left, out_right ), 1 )
            out = self.domain_classifier_layer.forward(out)

        return out

    def forward_one_side(self, batch):
        # batch is of size (len, batch_size)
        #sequence lengths of all lines in the text
        seq_lengths = [0 for _ in range(batch.size(1))]
        
        batch_cpu = batch.cpu().data.numpy()
        for i in range(batch_cpu.shape[0]):
            for j in range(batch_cpu.shape[1]):
                if batch_cpu[i][j] == 265728:
                    seq_lengths[j] = (i+1)

        #sorted sequence lengths
        seq_lengths_sorted = list(seq_lengths)
        sorted_index = sorted(range(len(seq_lengths)),key=lambda x:-seq_lengths[x])
        for j,val in enumerate(sorted_index):
            seq_lengths_sorted[j] = seq_lengths[val]
        original_index = list(sorted_index)

        #conver batch to "batch first"
        batch_permuted = batch.cuda().permute(1, 0).cuda()

        # batch sorted by length, required for packed padded sequence computations
        for ind,i in enumerate(sorted_index):
            original_index[i] = ind
        sorted_index = Variable(torch.LongTensor(sorted_index).cuda()).cuda()
        sorted_idx = sorted_index.view(-1, 1).expand(batch_permuted.size(0), batch_permuted.size(1)).cuda()
        batch_perm_sorted = batch_permuted.cuda().gather(0, sorted_idx).squeeze().cuda()

        emb = self.embedding(batch_perm_sorted.cuda())  # (len, batch_size, n_e)
        emb = Variable(emb.data.cuda()).cuda()

        idx_init = Variable(torch.LongTensor([x-1 for x in seq_lengths_sorted]).cuda()).cuda()
        odx_init = Variable(torch.LongTensor(original_index).cuda()).cuda()
        pack_wv = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths_sorted, batch_first=True)
        self.hidden = self.init_hidden(batch.size(1))
        out, (ht,ct) = self.lstm(pack_wv, self.hidden)

        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = idx_init.view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1).cuda()
        # this is the last state of the lstm
        decoded = unpacked.gather(1, idx).squeeze()
        
        # this is the lstm states in the original unsorted order
        odx = odx_init.view(-1, 1).expand(unpacked.size(0), unpacked.size(-1)).cuda()
        decoded = decoded.gather(0, odx)

        return decoded


    def __repr__(self):
        text = "SLC (\n{}\n{}\n)".format(
            indent(str(self.lstm), 2),
            indent(str(self.output_layer), 2)
        )
        return text


