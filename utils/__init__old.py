
import sys
import gzip
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_npz(path):
    data = np.load(path)
    return [ str(w) for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x


def pad(sequences, pad_token='<pad>', pad_left=False):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(len(seq) for seq in sequences)
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def make_batch(emblayer, sequences, oov='<oov>', pad_left=False):
    sequences = pad(sequences, pad_left=pad_left)
    batch_size = len(sequences)
    length = len(sequences[0])
    word2id, oovid = emblayer.word2id, emblayer.oovid
    data = torch.LongTensor(list(word2id.get(w, oovid) for s in sequences for w in s))
    assert data.size(0) == batch_size*length
    return data.view(batch_size, length).t().contiguous()


def pad_iter(corpus, emblayer, positive_iter, negative_iter, use_content=False, pad_left=False):
    batchify = lambda sequences: make_batch(emblayer, sequences, pad_left=pad_left)
    for pospairs, negpairs in zip(positive_iter, negative_iter):
        allpairs = [[],[]]
        allpairs[0] = pospairs[0] + negpairs[0] # left
        allpairs[1] = pospairs[1] + negpairs[1] # right
        labels = [1]*len(pospairs[0]) + [0]*len(negpairs[0])
        allpairs[0] = corpus.get(allpairs[0])
        allpairs[1] = corpus.get(allpairs[1])
        if not use_content:
            input_left, input_right = allpairs[0][0], allpairs[1][0]
            input_left, input_right = batchify(input_left), batchify(input_right)
        else:
            input_left = map(batchify, allpairs[0])
            input_right = map(batchify, allpairs[1])

        yield (input_left, input_right), torch.LongTensor(labels)


def cross_pad_iter(corpus, emblayer, positive_iter, negative_iter, cross_iter, use_content=False, pad_left=False):
    batchify = lambda sequences: make_batch(emblayer, sequences, pad_left=pad_left)
    for pospairs, negpairs, crosspairs in zip(positive_iter, negative_iter, cross_iter):
        task_allpairs = [[],[]]
        task_allpairs[0] = pospairs[0] + negpairs[0] # left
        task_allpairs[1] = pospairs[1] + negpairs[1] # right
        task_labels = [1]*len(pospairs[0]) + [0]*len(negpairs[0])
        task_allpairs[0] = corpus.get(task_allpairs[0])
        task_allpairs[1] = corpus.get(task_allpairs[1])
        if not use_content:
            input_left, input_right = task_allpairs[0][0], task_allpairs[1][0]
            input_left, input_right = batchify(input_left), batchify(input_right)
        else:
            input_left = map(batchify, task_allpairs[0])
            input_right = map(batchify, task_allpairs[1])

        domain_uids = crosspairs[0] + crosspairs[1]
        domain_labels = [0]*len(crosspairs[0]) + [1]*len(crosspairs[1])
        domain_input = corpus.get(domain_uids)
        if not use_content:
            domain_input = batchify(domain_input[0])
        else:
            domain_input = map(batchify, domain_input)

        yield (input_left, input_right), torch.LongTensor(task_labels), (domain_input), torch.LongTensor(domain_labels)


class FileLoader(object):
    def __init__(self, file_paths, batch_size, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.pos = 0
        data = []
        for file_path,key in file_paths:
            with open(file_path) as fin:
                curr_data = fin.readlines()
            data += [ tuple([y+key for y in x.split()[:2]]) for x in curr_data ]
        if shuffle:
            random.shuffle(data)
        self.tot = len(data)
        self.data_left = [ x[0] for x in data ]
        self.data_right = [ x[1] for x in data ]

    def __iter__(self):
        return self

    def next(self):
        pos, tot, batch_size = self.pos, self.tot, self.batch_size
        if pos < tot:
            self.pos += batch_size
            return (
                self.data_left[pos:pos+batch_size],
                self.data_right[pos:pos+batch_size]
            )
        else:
            raise StopIteration()

class RandomLoader(object):
    def __init__(self, corpus, exclusive_set, batch_size):
        self.uids = corpus.uids()
        self.N = len(self.uids)
        self.exclusive_set = set(exclusive_set)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        N, batch_size = self.N, self.batch_size
        uids, exclusive_set = self.uids, self.exclusive_set
        cnt = 0
        batch_left = []
        batch_right = []
        while cnt < batch_size:
            x = uids[random.randint(0, N-1)]
            y = uids[random.randint(0, N-1)]
            if (x!=y) and ((x,y) not in exclusive_set) and ((y,x) not in exclusive_set):
                cnt += 1
                batch_left.append(x)
                batch_right.append(y)
        return (batch_left, batch_right)


class TwoDomainLoader(object):
    def __init__(self, domain1_paths, domain2_paths, batch_size):
        self.uids_1 = self.read_domain(domain1_paths)
        self.uids_2 = self.read_domain(domain2_paths)
        self.batch_size = batch_size

    def read_domain(self, file_paths):
        data = [ ]
        for file_path,key in file_paths:
            with gzip.open(file_path) as fin:
                curr_data = fin.readlines()
            data += [ x.split()[0]+key for x in curr_data ]
        return data

    def __iter__(self):
        return self

    def next(self):
        uids_1, uids_2, batch_size = self.uids_1, self.uids_2, self.batch_size
        batch_left = random.sample(uids_1, batch_size)
        batch_right = random.sample(uids_2, batch_size)
        return (batch_left, batch_right)


class CombinedLoader(object):
    def __init__(self, loader_1, loader_2, batch_size):
        k = batch_size / 2
        assert batch_size%2 == 0
        loader_1.batch_size = k
        loader_2.batch_size = k
        self.loader_1 = loader_1
        self.loader_2 = loader_2

    def __iter__(self):
        return self

    def next(self):
        bl1, br1 = self.loader_1.next()
        bl2, br2 = self.loader_2.next()
        return (bl1+bl2, br1+br2)

class EmbeddingLayer(object):
    def __init__(self, n_d, words, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', dictionary=None):
        word2id = {}
        emb_words = []
        emb_vecs = np.array([])
        if embs is not None:
            embwords, embvecs = embs
            for ind,word in enumerate(embwords):
                if word not in dictionary:
                    continue
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)
                emb_words.append(word)
                numpy.append(emb_vecs, embvecs[ind])

            embvecs = emb_vecs
            embwords = emb_words

            say("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            if n_d != len(embvecs[0]):
                say("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            say("embedding shape: {}\n".format(weight.size()))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def set_idf_embeddings(self, file_paths, n_d, words, embs=None, fix_emb=True, oov='<oov>', pad='<pad>'):

        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)
        lst = []
        for file_path in file_paths:
            with gzip.open(file_path) as fin:
                for line in fin:
                    parts = line.split('\t')
                    uid, title = parts[0], parts[1]
                    content = parts[2] if len(parts)>2 else ""
                    lst.append(title)
                    lst.append(content)
        vectorizer.fit_transform(lst)
        word_idf = {}
        idfs = vectorizer.idf_
        avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
        for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
            word_idf[word] = idf_value
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            embvecs = np.copy(embvecs)
            for ind,word in enumerate(embwords):
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)
                if word in word_idf:
                    embvecs[ind] *= word_idf[word]
                else:
                    embvecs[ind] *= avg_idf

            #say("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            if n_d != len(embvecs[0]):
            #    say("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
            #        n_d, len(embvecs[0]), len(embvecs[0])
            #    ))
                n_d = len(embvecs[0])

        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        #model.embedding = nn.Embedding(model.n_V, n_d)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            #say("embedding shape: {}\n".format(weight.size()))
            self.embedding.weight = weight

        if fix_emb:
            self.embedding.weight.requires_grad = False


class Corpus(object):
    def __init__(self, file_paths, bos='<s>', eos='</s>'):
        self.file_paths = file_paths
        self.data_title = {}
        self.data_content = {}
        self.words = set()
        for file_path,key in file_paths:
            with gzip.open(file_path) as fin:
                for line in fin:
                    parts = line.split('\t')
                    uid, title = parts[0]+key, parts[1]
                    content = parts[2] if len(parts)>2 else ""
                    title = [bos] + title.split() + [eos]
                    content = [bos] + content.split() + [eos]
                    for word in title[1:-1]+content[1:-1]:
                        self.words.add(word)
                    self.data_title[uid] = title
                    self.data_content[uid] = content

    def get(self, uids):
        return self.get_title(uids), self.get_content(uids)

    def get_title(self, uids):
        if isinstance(uids, list) or isinstance(uids, tuple):  # multiple keys
            return [ self.data_title[x] for x in uids ]
        else:  # single key
            return self.data_title[uids]

    def get_content(self, uids):
        if isinstance(uids, list) or isinstance(uids, tuple):  # multiple keys
            return [ self.data_content[x] for x in uids ]
        else:  # single key
            return self.data_content[uids]

    def keys(self):
        return self.data_title.keys()

    def uids(self):
        return self.data_title.keys()

    def docs(self):
        for uid, title in self.data_title.items():
            content = self.data_content[uid]
            yield title, content
