'''
Training for background dataset resampling
WIP: work not completed

'''

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchtext
from torchtext import data
from torchtext import datasets

from utils.trainer import train_epoch_resample, eval_epoch_dual

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], type=int, nargs='+')

# model info
parser.add_argument('--load-path', type=str)

# dataset info
parser.add_argument('-id', '--in-dataset',type=str, choices=['sst', '20ng', 'trec'], default='sst')
parser.add_argument('-od', '--out-dataset', type=str, choices=['wikitext2', 'wikitext103', 'gutenberg'], default=None)
parser.add_argument('--scale', default=32, type=int)
parser.add_argument('--crop', default=32, type=int)
parser.add_argument('--no-flip', action='store_true')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-j', '--workers', default=8, type=int)

# optimization
parser.add_argument('--maxent', action='store_true')
parser.add_argument('--coef', default=0.5, type=float)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr-w', default=100, type=float)
parser.add_argument('--lr-step', default=30, type=int)
parser.add_argument('--lr-gamma', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

# evaluation
parser.add_argument('--topk', default=[1, 5], type=int, nargs='+')
parser.add_argument('--print-freq', default=-1, type=int)
parser.add_argument('--test-freq', default=1, type=int)
parser.add_argument('--save-freq', default=10, type=int)


# get available gpus
args = parser.parse_args()
if args.gpus[0] < 0:
    import GPUtil
    n_gpus = -args.gpus[0] if args.gpus[0] < -1 else 4
    args.gpus = [int(i) for i in GPUtil.getAvailable(order='first', limit=n_gpus, maxMemory=0.15)]
    if len(args.gpus) < n_gpus:
        raise RuntimeError('No enough GPUs')
    print('Using GPUs:', *args.gpus)
torch.cuda.set_device(args.gpus[0])


if args.in_dataset == 'sst':
    # set up fields
    TEXT = data.Field(pad_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=False, train_subtrees=False,
        filter_pred=lambda ex: ex.label != 'neutral')

    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))

    # create our own iterator, avoiding the calls to build_vocab in SST.iters
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, repeat=False)
    n_class = 2

elif args.in_dataset == '20ng':
    TEXT = data.Field(pad_first=True, lower=True, fix_length=100)
    LABEL = data.Field(sequential=False)

    train = data.TabularDataset(path='./.data/20newsgroups/20ng-train.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    test = data.TabularDataset(path='./.data/20newsgroups/20ng-test.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))
    
    train_iter = data.BucketIterator(train, batch_size=args.batch_size, repeat=False)
    test_iter = data.BucketIterator(test, batch_size=args.batch_size, repeat=False)
    n_class = 20

elif args.in_dataset == 'trec':
    # set up fields
    TEXT = data.Field(pad_first=True, lower=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)


    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))
    print('num labels:', len(LABEL.vocab))

    # make iterators
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, repeat=False)
    n_class = 50 


if args.out_dataset == 'wikitext2':
    path = './.data/wikitext_reformatted/{}_sentences'.format(args.out_dataset)
    path = path+f"_{n_class}"

    # path = './.data/wikitext_reformatted/wikitext2_sentences'
    # with open(path) as s:  # script helpful for preprocessing ood dataset.
    #     path = path+f"_{n_class}"
    #     with open(path,'w') as d:
    #         for s_line in s:
    #             d.write(f'{n_class}, {s_line}')
        
    TEXT_custom = data.Field(pad_first=False, lower=True)
    LABEL_custom = data.Field(sequential=False, use_vocab=False) #for reading what's in file 
    custom_data = data.TabularDataset(path=path,
                                    format='csv', 
                                    fields=[('label', LABEL_custom), ('text', TEXT_custom)])
    TEXT_custom.build_vocab(train.text, max_size=10000)
    import pdb; pdb.set_trace()
    #LABEL_custom.build_vocab(np.sort(np.unique(list(train.label) + list(custom_data.label)).astype('int')).astype('str'))
    print('vocab length (including special tokens):', len(TEXT_custom.vocab))

    train_iter_oe = data.BucketIterator(custom_data, batch_size=args.batch_size, repeat=False)

    n_class = n_class + 1

   
elif args.out_dataset == 'wikitext103': # TODO prepare the preprocessed data like wikitext2
    TEXT_custom = data.Field(pad_first=True, lower=True)

    custom_data = data.TabularDataset(path='./.data/wikitext_reformatted/wikitext103_sentences',
                                      format='csv',
                                      fields=[('text', TEXT_custom)])

    TEXT_custom.build_vocab(train.text, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT_custom.vocab))

    train_iter_oe = data.BucketIterator(custom_data, batch_size=args.batch_size, repeat=False)
    n_class = n_class + 1


elif args.out_dataset == 'gutenberg':
    TEXT_custom = data.Field(pad_first=True, lower=True)

    custom_data = data.TabularDataset(path='./.data/gutenberg/gutenberg_sentences',
                                      format='csv',
                                      fields=[('text', TEXT_custom)])

    TEXT_custom.build_vocab(train.text, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT_custom.vocab))

    train_iter_oe = data.BucketIterator(custom_data, batch_size=args.batch_size, repeat=False)
    n_class = n_class + 1


elif args.out_dataset is None: # by default is OFF
    train_iter_oe = [None]

else:
    print("Unsupported OoD Dataset")
    sys.exit(0)


class ClfGRU(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2,
            bias=True, batch_first=True,bidirectional=False)
        self.linear = nn.Linear(128, n_class)

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(hidden)
        return logits

model = ClfGRU(n_class).to(device)

print("Number of classes:",n_class)


if args.load_path:
    model.load_state_dict(torch.load(args.load_path))
model = torch.nn.DataParallel(model, args.gpus).cuda()
torch.backends.cudnn.benchmark = True

weight_params = nn.Parameter(torch.zeros(len(train_out_dataset)).cuda(args.gpus[0]))

# optimization
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), args.lr)
else:
    raise ValueError('Invalid optimizer!')
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)

optimizer_w = optim.SGD([weight_params], args.lr_w, momentum=0.9)

# uniformity loss
def weighted_entropy_loss(logits, weights=None, coef=args.coef):
    scores = F.softmax(logits, 1)
    log_scores = F.log_softmax(logits, 1)
    if weights is None:
        if args.maxent:
            loss = torch.sum(scores * log_scores) / logits.size(0) + np.log(n_class)  # maximize output entropy
        else:
            loss = torch.mean(-log_scores) - np.log(n_class)  # minimize kl div to uniform
    else:
        if args.maxent:
            loss = torch.sum(weights @ (scores * log_scores)) * len(train_out_dataset) / logits.size(0) + np.log(n_class)
        else:
            kld = -log_scores.mean(1) - np.log(n_class)
            loss = torch.mean(weights * kld)
    return coef * loss

# start training
for epoch in range(1, args.epochs + 1):
    w = F.softplus(weight_params).detach().cpu()
    out_sampler = torch.utils.data.WeightedRandomSampler(w ** .5, len(train_out_dataset), replacement=True)
    train_out_loader = DataLoader(train_out_dataset, batch_size=args.batch_size, shuffle=False, sampler=out_sampler, num_workers=args.workers, pin_memory=False, drop_last=True)
    train_out_loader = iter(cycle(train_out_loader))
    
    train_epoch_resample(epoch, train_in_loader, train_out_loader, weight_params, model, weighted_entropy_loss, optimizer, optimizer_w, scheduler, args)
    if epoch % args.test_freq == 0:
        loss, acc = eval_epoch_dual(epoch, val_in_loader, val_out_loader, model, weighted_entropy_loss, args)
    if epoch % args.save_freq == 0:
        save_name = args.in_dataset + '_' + args.out_dataset + '_' + args.arch + '_resample'
        save_path = os.path.join('checkpoints/', save_name + '_{}ep-{:04d}top{}.pth'.format(epoch, round(acc[0] * 10000), args.topk[0]))
        torch.save(model.module.state_dict(), save_path)
        w_path = os.path.join('checkpoints/', save_name + '_{}ep-weights.pth'.format(epoch))
        torch.save(weight_params.detach().cpu(), w_path)

for i, k in enumerate(args.topk):
    print('Top {} Accuracy = {:.2%}'.format(k, acc[i]))