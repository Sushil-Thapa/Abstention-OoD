'''
training for in-distribution classification + out-of-distribution detection
'''

import os
import sys
import argparse
from functools import reduce
import copy
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

from nltk import ngrams
from itertools import combinations


import datasets

dset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

sys.path.append('.')
sys.path.append('/home/zeus/lanl/ood/code/dac-ood/')


parser = argparse.ArgumentParser()

# dataset info
parser.add_argument('-od', '--out-dataset', default='tiny_images', choices=dset_names)
parser.add_argument('--scale', default=32, type=int)
parser.add_argument('--crop', default=32, type=int)

# resampling
parser.add_argument('--resample', type=str,  default='random')
parser.add_argument('-p', '--ratio', default=0.1, type=float)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('-r', '--repeat', default=5, type=int)

# get available gpus
args = parser.parse_args()
print(args)

# data loading
normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.25] * 3)
flip_prob = 0.5

pad = args.crop // 8
train_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.RandomCrop(args.crop, padding=pad),
    transforms.RandomHorizontalFlip(flip_prob),
    transforms.ToTensor(),
    normalize
])

train_out_dataset, _ = datasets.__dict__[args.out_dataset](train=True, transform=train_transform)
full_train_out_dataset = copy.deepcopy(train_out_dataset)

keep_idxs = []

if args.resample:
    print(args.resample)
    if args.resample == 'random':
        for _ in range(args.repeat):
            w = torch.rand(len(train_out_dataset))
            w_thresh = torch.kthvalue(w, int((1 - args.ratio) * len(w)))[0]
            keep_idx = [c.item() for c in (w >= w_thresh).nonzero()]
            keep_idxs.append(keep_idx)
    else:
        import glob
        models = glob.glob(f"{args.resample}/*.pth")
        print(models)
            
        # import pdb; pdb.set_trace()

        for model in models:
            w = torch.load(model, map_location=lambda storage, loc: storage).detach()
            p = F.softplus(w)
            p = args.ratio * p / p.mean()
            keep_idx = [c.item() for c in (p >= torch.rand(len(w))).nonzero()]
            keep_idxs.append(keep_idx)
            

    # print('Use OOD examples: {}/{} ({:.2%})'.format(len(keep_idx), len(train_out_dataset), len(keep_idx) / len(train_out_dataset)))
    # train_out_dataset = Subset(train_out_dataset, keep_idx)


repeat = args.repeat

idx_ngrams = []
for i in range(repeat):
    if i > 0:
        idx_ngrams.append(list(combinations(range(repeat), i+1)))
print(idx_ngrams)

    

# df = pd.DataFrame(idx_ngrams)
# a = df[~df.isnull()].iloc[1].apply(lambda x: [keep_idxs[i] for i in x])

# sns_data = {}
# sns_data['num of samples'] = []
# sns_data['mean IOU %'] = []



ngram_indices = []
n_scores = {}
for idx_ngram in idx_ngrams: 
    # [(1,2), (2,3)]
    n = len(idx_ngram[0])
    # print('idx_ngram',n, idx_ngram)

    x = []  # mean, std
    for ngram in idx_ngram: 
        # (1,2)
        # print('ngram',ngram)
        ngram = [keep_idxs[i] for i in ngram]
        intersection = reduce(np.intersect1d, ngram)
        union = reduce(np.union1d, ngram)

        iou = len(intersection) / len(union) * 100
        # print(n, iou*100)
        x.append(iou)

        # sns_data['num of samples'].append(n)
        # sns_data['mean IOU %'].append(round(iou,4))

    
    n_scores[n] = {
        'mean IOU %': round(np.array(x).mean(),4),
        'std%': round(np.array(x).std(),4),
        'n-gram':n,
        'IOU':[round(i,4) for i in x]
    }
    # print({n:x})

    

# print(n_scores)
import seaborn as sns
df = pd.DataFrame.from_dict(n_scores, orient='index')

fig = plt.figure()
plt.errorbar(df['n-gram'], df['mean IOU %'], df["std%"],fmt='-o')

# fig.suptitle('Uniform Random Samples (topk) IoU', fontsize=20)
fig.suptitle('Adversarial resampling (topk) IoU', fontsize=20)
plt.xlabel('num of samplers compared', fontsize=13)
plt.ylabel('mean IOU %', fontsize=12)
filename = 'random' if args.resample == 'random' else "adversarial"
fig.savefig(f"assets/{filename}.png")

plt.show()

# sns_plot = sns.lineplot(data=df, x = 'n-gram', y='mean IOU %')
# sns_plot.figure.savefig(f"assets/{args.resample}.png")
# import pdb; pdb.set_trace()

print('')
# ngram_indices.append([[[keep_idxs[i], keep_idxs[j]] for i in ngram for j in i])

# ijk according to len


# train_out_loader = DataLoader(train_out_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
# intersect = reduce(np.intersect1d, keep_idxs)
# union = reduce(np.union1d, keep_idxs)

# import pdb; pdb.set_trace()

