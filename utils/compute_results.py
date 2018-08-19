import math
import numbers
import numpy as np
import torch

class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass

class IRMeter(Meter):

    def __init__(self):
        super(IRMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()
        self.data = [ ]

    def add(self, output, target, data_pairs):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        for d in data_pairs:
            self.data.append(d)
        self.sortind = None

    def value(self, max_fpr=1.0):

        question_followups = {}
        pair_score = {}
        pair_label = {}
        for ind,pair in enumerate(list(self.data)):
            if tuple(pair) in pair_score:
                continue
            if pair[0] in question_followups:
                question_followups[pair[0]].append(pair[1])
            else:
                question_followups[pair[0]] = [pair[1]]
            pair_score[tuple(pair)] = self.scores[ind]
            pair_label[tuple(pair)] = self.targets[ind]

        assert len(self.data) == len(pair_score)

        mrr = 0.0
        for question in question_followups:
            follow_ups = question_followups[question]
            arr = [ ]
            labels = [ ]
            current_score = 0.0
            for fw in follow_ups:
                arr.append(pair_score[tuple([question,fw])])
                labels.append(int(pair_label[tuple([question,fw])]))
            assert len(follow_ups) == 11
            assert len(arr) == 11
            current_score = arr[labels.index(1)]
            arr = sorted(arr, reverse=True)
            assert current_score in arr
            rank = arr.index(current_score) + 1
            mrr += 1.0/(rank)

        return mrr/len(question_followups)
