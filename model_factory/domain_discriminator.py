
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(10)

class ClassificationD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(ClassificationD, self).__init__()
        self.n_in = encoder.n_out
        self.n_out = 300

        self.seq = nn.Sequential(
            nn.Linear(self.n_in, self.n_out),
            nn.ReLU(),
            nn.Linear(self.n_out, self.n_out),
            nn.ReLU(),
            nn.Linear(self.n_out, 2)
        )

    def forward(self, x):
        return self.seq(x)

    def compute_loss(self, output, labels):
        return nn.functional.cross_entropy(output, labels)

class WassersteinD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(WassersteinD, self).__init__()
        self.n_in = encoder.n_out
        self.n_out = 300

        self.seq = nn.Sequential(
            nn.Linear(self.n_in, self.n_out),
            nn.ReLU(),
            nn.Linear(self.n_out, self.n_out),
            nn.ReLU(),
            nn.Linear(self.n_out, 1)
        )

    def forward(self, x):
        self.clip_weights()
        return self.seq(x)

    def clip_weights(self, val_range=0.01):
        for p in self.parameters():
            p.data.clamp_(min=-val_range, max=val_range)

    def compute_loss(self, output, labels):
        #print output.data.size()
        #print labels.data.size()
        assert output.size(1) == 1
        assert output.size(0) == labels.size(0)
        assert output.size(0) % 2 == 0
        labels = labels.float()*2-1
        #print labels
        #print labels.data.size()
        #print output*labels
        #print (output*labels).mean()
        loss = (output*labels).mean()
        return loss
