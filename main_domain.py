
import sys
import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboard import summary
from tensorboard import FileWriter

from model_factory import get_model_class
from model_factory.domain_discriminator import ClassificationD, WassersteinD
from utils import Corpus, EmbeddingLayer, FileLoader, TwoDomainLoader
from utils import say, load_embedding, pad_iter, make_batch, cross_pad_iter
from utils.meter import AUCMeter

def train(iter_cnt, model, domain_d, corpus, args, optimizer_encoder, optimizer_domain_d):

    train_writer = FileWriter(args.run_path+"/train", flush_secs=5)

    pos_file_path = "{}.pos.txt".format(args.train)
    neg_file_path = "{}.neg.txt".format(args.train)

    train_corpus_path = os.path.dirname(args.train) + "/corpus.tsv.gz"
    cross_train_corpus_path = os.path.dirname(args.cross_train) + "/corpus.tsv.gz"

    use_content = False
    if args.use_content:
        use_content = True

    pos_batch_loader = FileLoader(
        [ tuple([pos_file_path, os.path.dirname(args.train)]) ],
        args.batch_size
    )
    neg_batch_loader = FileLoader(
        [ tuple([neg_file_path, os.path.dirname(args.train)]) ],
        args.batch_size
    )
    cross_loader = TwoDomainLoader(
        [ tuple([train_corpus_path, os.path.dirname(train_corpus_path)]) ],
        [ tuple([cross_train_corpus_path, os.path.dirname(cross_train_corpus_path)]) ],
        args.batch_size*2
    )

    embedding_layer = model.embedding_layer

    criterion1 = model.compute_loss
    criterion2 = domain_d.compute_loss

    start = time.time()
    task_loss = 0.0
    task_cnt = 0
    domain_loss = 0.0
    dom_cnt = 0
    total_loss = 0.0
    total_cnt = 0

    for batch, labels, domain_batch, domain_labels in cross_pad_iter(corpus, embedding_layer, pos_batch_loader,
                neg_batch_loader, cross_loader, use_content, pad_left=False):
        iter_cnt += 1

        new_batch = [ ]
        if args.use_content:
            for x in batch:
                for y in x:
                    new_batch.append(y)
            batch = new_batch
            domain_batch = [ x for x in domain_batch ]

        if args.cuda:
            batch = [ x.cuda() for x in batch ]
            labels = labels.cuda()
            if not use_content:
                domain_batch = domain_batch.cuda()
            else:
                domain_batch = [ x.cuda() for x in domain_batch ]
            domain_labels = domain_labels.cuda()
        batch = map(Variable, batch)
        labels = Variable(labels)
        if not use_content:
            domain_batch = Variable(domain_batch)
        else:
            domain_batch = map(Variable, domain_batch)
        domain_labels = Variable(domain_labels)

        model.zero_grad()
        domain_d.zero_grad()

        repr_left = None
        repr_right = None
        if not use_content:
            repr_left = model(batch[0])
            repr_right = model(batch[1])
        else:
            repr_left = model(batch[0]) + model(batch[1])
            repr_right = model(batch[2]) + model(batch[3])
        output = model.compute_similarity(repr_left, repr_right)
        loss1 = criterion1(output, labels)
        task_loss += loss1.data[0]*output.size(0)
        task_cnt += output.size(0)

        domain_output = None
        if not use_content:
            domain_output = domain_d(model(domain_batch))
        else:
            domain_output = domain_d(model(domain_batch[0])) + domain_d(model(domain_batch[1]))
        loss2 = criterion2(domain_output, domain_labels)
        domain_loss += loss2.data[0]*domain_output.size(0)
        dom_cnt += domain_output.size(0)

        loss = loss1 - args.lambda_d*loss2
        total_loss += loss.data[0]
        total_cnt += 1
        loss.backward()
        optimizer_encoder.step()
        optimizer_domain_d.step()

        if iter_cnt % 100 == 0:
            say("\r" + " "*50)
            say("\r{} tot_loss: {:.4f} task_loss: {:.4f} domain_loss: {:.4f} eps: {:.0f} ".format(
                iter_cnt, total_loss/total_cnt, task_loss / task_cnt, domain_loss / dom_cnt,
                (task_cnt + dom_cnt)/(time.time()-start)
            ))
            s = summary.scalar('loss', total_loss/total_cnt)
            train_writer.add_summary(s, iter_cnt)

    say("\n")
    train_writer.close()

    return iter_cnt

def evaluate(iter_cnt, filepath, model, corpus, args, logging=True):
    if logging:
        valid_writer = FileWriter(args.run_path+"/valid", flush_secs=5)

    pos_file_path = "{}.pos.txt".format(filepath)
    neg_file_path = "{}.neg.txt".format(filepath)
    pos_batch_loader = FileLoader(
        [tuple([pos_file_path, os.path.dirname(args.eval)])],
        args.batch_size
    )
    neg_batch_loader = FileLoader(
        [tuple([neg_file_path, os.path.dirname(args.eval)])],
        args.batch_size
    )

    batchify = lambda bch: make_batch(model.embedding_layer, bch)
    model.eval()
    criterion = model.compute_loss
    auc_meter = AUCMeter()
    scores = [ np.asarray([], dtype='float32') for i in range(2) ]
    for loader_id, loader in enumerate((neg_batch_loader, pos_batch_loader)):
        for data in loader:
            data = map(corpus.get, data)
            batch = None
            if not args.eval_use_content:
                batch = (batchify(data[0][0]), batchify(data[1][0]))
            else:
                batch = (map(batchify, data[0]), map(batchify, data[1]))
                new_batch = [ ]
                for x in batch:
                    for y in x:
                        new_batch.append(y)
                batch = new_batch
            labels = torch.ones(batch[0].size(1)).type(torch.LongTensor)*loader_id
            if args.cuda:
                batch = [ x.cuda() for x in batch ]
                labels = labels.cuda()
            if not args.eval_use_content:
                batch = (Variable(batch[0], volatile=True), Variable(batch[1], volatile=True))
            else:
                batch = (Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True), Variable(batch[3], volatile=True))
            labels = Variable(labels)
            if not args.eval_use_content:
                repr_left = model(batch[0])
                repr_right = model(batch[1])
            else:
                repr_left = model(batch[0]) + model(batch[1])
                repr_right = model(batch[2]) + model(batch[3])
            output = model.compute_similarity(repr_left, repr_right)

            if model.criterion.startswith('classification'):
                assert output.size(1) == 2
                output = nn.functional.log_softmax(output)
                current_scores = -output[:,loader_id].data.cpu().squeeze().numpy()
                output = output[:,1]
            else:
                assert output.size(1) == 1
                current_scores = output.data.cpu().squeeze().numpy()
            auc_meter.add(output.data, labels.data)
            scores[loader_id] = np.append(scores[loader_id], current_scores)

    auc_score = auc_meter.value()
    auc10_score = auc_meter.value(0.1)
    auc05_score = auc_meter.value(0.05)
    auc02_score = auc_meter.value(0.02)
    auc01_score = auc_meter.value(0.01)
    if model.criterion.startswith('classification'):
        avg_score = (scores[1].mean()+scores[0].mean())*0.5
    else:
        avg_score = scores[1].mean()-scores[0].mean()
    say("\r[{}] auc(.01): {:.3f}  auc(.02): {:.3f}  auc(.05): {:.3f}"
            "  auc(.1): {:.3f}  auc: {:.3f}"
            "  scores: {:.2f} ({:.2f} {:.2f})\n".format(
        os.path.basename(filepath).split('.')[0],
        auc01_score,
        auc02_score,
        auc05_score,
        auc10_score,
        auc_score,
        avg_score,
        scores[1].mean(),
        scores[0].mean()
    ))

    if logging:
        s = summary.scalar('auc', auc_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.1)', auc10_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.05)', auc05_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.02)', auc02_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.01)', auc01_score)
        valid_writer.add_summary(s, iter_cnt)
        valid_writer.close()

    return auc05_score

def main(args):
    model_class = get_model_class(args.model)
    model_class.add_config(argparser)
    ClassificationD.add_config(argparser)
    args, _ = argparser.parse_known_args()
    say(args)

    args.run_id = random.randint(0,10**9)
    args.run_path = "{}/{}".format(args.run_dir, args.run_id)
    #if not os.path.exists(args.run_dir):
    #    os.makedirs(args.run_dir)
    #assert os.path.isdir(args.run_dir)
    #assert not os.path.exists(args.run_path)
    #os.makedirs(args.run_path)
    say("\nRun ID: {}\nRun Path: {}\n\n".format(
        args.run_id,
        args.run_path
    ))


    train_corpus_path = os.path.dirname(args.train) + "/corpus.tsv.gz"
    cross_train_corpus_path = os.path.dirname(args.cross_train) + "/corpus.tsv.gz"
    train_corpus = Corpus([ tuple([train_corpus_path,os.path.dirname(args.train)]), tuple([cross_train_corpus_path,os.path.dirname(args.cross_train)]) ])
    valid_corpus_path = os.path.dirname(args.eval) + "/corpus.tsv.gz"
    valid_corpus = Corpus([ tuple([valid_corpus_path,os.path.dirname(args.eval)]) ])
    say("Corpus loaded.\n")

    embs = load_embedding(args.embedding) if args.embedding else None

    embedding_layer = EmbeddingLayer(args.n_d, ['<s>', '</s>'],
        embs
    )

    model = model_class(
        embedding_layer,
        args
    )
    if args.wasserstein:
        domain_d = WassersteinD(
            model,
            args
        )
        args.lambda_d = -args.lambda_d
    else:
        domain_d = ClassificationD(
            model,
            args
        )

    if args.cuda:
        model.cuda()
        domain_d.cuda()
    say("\n{}\n\n".format(model))
    say("\n{}\n\n".format(domain_d))

    needs_grad = lambda x: x.requires_grad

    optimizer_encoder = optim.Adam(
        filter(needs_grad, model.parameters()),
        lr = args.lr
    )
    optimizer_domain_d = optim.Adam(
        filter(needs_grad, domain_d.parameters()),
        lr = -args.lr2
    )

    best_dev = 0
    iter_cnt = 0
    for epoch in range(args.max_epoch):
        iter_cnt = train(iter_cnt, model, domain_d, train_corpus, args, optimizer_encoder, optimizer_domain_d)
        current_dev = evaluate(iter_cnt, args.eval+"/dev", model, valid_corpus, args)
        if current_dev > best_dev:
            best_dev = current_dev
            evaluate(iter_cnt, args.eval+"/test", model, valid_corpus, args, False)
        say("\n")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model) 


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--run_dir",  type=str, default="/D/home/tao/mnt/ASAPPNAS/tao/test")
    argparser.add_argument("--model", type=str, required=True, help="which model class to use")
    argparser.add_argument("--embedding", "--emb", type=str, help="path of embedding")
    argparser.add_argument("--train", type=str, required=True, help="training file")
    argparser.add_argument("--wasserstein", action="store_true")
    argparser.add_argument("--cross_train", type=str, required=True, help="cross training file")
    argparser.add_argument("--eval", type=str, required=True, help="validation file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=100)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--learning", type=str, default='adam')
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr2", type=float, default=0.0001)
    argparser.add_argument("--lambda_d", type=float, default=0.01)
    argparser.add_argument("--use_content", action="store_true")
    argparser.add_argument("--eval_use_content", action="store_true")
    argparser.add_argument("--save_model", type=str, required=False, help="location to save model")

    args, _  = argparser.parse_known_args()
    main(args)
