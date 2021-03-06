'''
Training for background dataset resampling
'''

import os
import sys
import argparse
import numpy as np
'''
TODO: ood dataset class is 0 from tinyimages.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import datasets
from utils.trainer import train_epoch_resample, eval_epoch_dual

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

sys.path.append('.')
sys.path.append('/home/zeus/lanl/ood/code/dac-ood/')


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], type=int, nargs='+')

# model info
parser.add_argument('-a', '--arch', default='wrn40', choices=model_names)
parser.add_argument('--load-path', type=str)

# dataset info
parser.add_argument('-id', '--in-dataset', default='cifar10', choices=dset_names)
parser.add_argument('-od', '--out-dataset', default='imagenet', choices=dset_names)
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

# reproducibility
parser.add_argument('--seed', default = 42, type=int)

args = parser.parse_args()

os.environ['PYTHONHASHSEED']=str(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# get available gpus

if args.gpus[0] < 0:
    import GPUtil
    n_gpus = -args.gpus[0] if args.gpus[0] < -1 else 4
    args.gpus = [int(i) for i in GPUtil.getAvailable(order='first', limit=n_gpus, maxMemory=0.15)]
    if len(args.gpus) < n_gpus:
        raise RuntimeError('No enough GPUs')
    print('Using GPUs:', *args.gpus)
torch.cuda.set_device(args.gpus[0])

# data loading
normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.25] * 3)  # TODOsu update normalization to match?
if args.in_dataset in ['mnist', 'svhn'] and not args.no_flip:
    print('Horizontal flip disabled for', args.in_dataset)
    args.no_flip = True
flip_prob = 0 if args.no_flip else 0.5

pad = args.crop // 8
train_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.RandomCrop(args.crop, padding=pad),
    transforms.RandomHorizontalFlip(flip_prob),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.CenterCrop(args.crop),
    transforms.ToTensor(),
    normalize
])

train_in_dataset, n_class = datasets.__dict__[args.in_dataset](train=True, transform=train_transform)
train_out_dataset, _ = datasets.__dict__[args.out_dataset](train=True, transform=train_transform)
val_in_dataset, _ = datasets.__dict__[args.in_dataset](train=False, transform=val_transform)
val_out_dataset, _ = datasets.__dict__[args.out_dataset](train=False, transform=val_transform)

# training set returns indices
train_out_dataset = datasets.IndexedDataset(train_out_dataset)

train_in_loader = DataLoader(train_in_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
train_out_loader = DataLoader(train_out_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)

val_in_loader = DataLoader(val_in_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
val_out_loader = DataLoader(val_out_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_out_loader = iter(cycle(train_out_loader))
val_out_loader = iter(cycle(val_out_loader))

# create model
n_class = n_class + 1 # Abstention Class
model = models.__dict__[args.arch](n_class)
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
    w = F.softplus(weight_params).detach().cpu() # activation func
    out_sampler = torch.utils.data.WeightedRandomSampler(w ** .5, len(train_out_dataset), replacement=True)  # sqrt normalize?
    train_out_loader = DataLoader(train_out_dataset, batch_size=args.batch_size, shuffle=False, sampler=out_sampler, num_workers=args.workers, pin_memory=False, drop_last=True)
    train_out_loader = iter(cycle(train_out_loader))
    
    train_epoch_resample(epoch, train_in_loader, train_out_loader, weight_params, model, weighted_entropy_loss, optimizer, optimizer_w, scheduler, args)
    if epoch % args.test_freq == 0:
        loss, acc = eval_epoch_dual(epoch, val_in_loader, val_out_loader, model, weighted_entropy_loss, args)
    if epoch % args.save_freq == 0:
        save_name = args.in_dataset + '_' + args.out_dataset + '_' + args.arch + '_resample'
        save_path = os.path.join('checkpoints/', save_name + '_{}ep-{:04d}top{}.pth'.format(epoch, round(acc[0] * 10000), args.topk[0]))
        torch.save(model.module.state_dict(), save_path)
        w_path = os.path.join('checkpoints/', save_name + '_seed{}_{}ep-weights.pth'.format(args.seed, epoch))
        torch.save(weight_params.detach().cpu(), w_path)

for i, k in enumerate(args.topk):
    print('Top {} Accuracy = {:.2%}'.format(k, acc[i]))