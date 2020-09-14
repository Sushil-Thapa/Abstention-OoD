'''
Training for background dataset resampling
'''
'''
thoughts
added train and val out dataset and loaders
added and removed idexes from dataset.
todo change out dataset channel to 1 instead of 3
change training steps.
'''
import os
import sys
import argparse
import numpy as np

import copy

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

from networks import config as cf

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], type=int, nargs='+')

# model info
parser.add_argument('--load-path', type=str)

# dataset info
parser.add_argument('-id', '--in-dataset',type=str, help='dataset = [mnist/cifar10/cifar100/stl10-labeled/fashion]', default='mnist')
parser.add_argument('-od', '--out-dataset', type=str, help='dataset=[tiny-images/stl10-unlabeled]This will be assigned to an extra class', default=None)
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
parser.add_argument('--seed', default=42, type=int)

#hyperparameters
parser.add_argument('--dropout', default=0.2, type=float, help='dropout_rate')


#cv dac
parser.add_argument('--datadir',  type=str, required=True, help='data directory')
parser.add_argument('--ood_train_x', default=None, type=str, help='Location for OoD data set')


args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
import torchvision
import torchvision.transforms as transforms


import os
import sys
import time
import datetime

from torch.autograd import Variable
import gpu_utils
import pdb
import numpy as np
from networks import wide_resnet,lenet,vggnet, resnet, resnet2
from networks import config as cf
import math


print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_device = device

start_epoch, num_epochs = 0, args.epochs
best_acc = 0.

def get_stl10_num_classes(train_y):
    ty = np.fromfile(train_y,dtype=np.uint8)
    return np.max(ty) + 1

print('\n[Phase 1] : Data Preparation')

### CIFAR-10/100 Transforms
if args.in_dataset == 'cifar10' or args.in_dataset == 'cifar100':
    cifar_mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
    }

    cifar_std = {
        #'cifar10': (0.2023, 0.1994, 0.2010),
        #below for evaluating manifold mixup models that were trained using the following std vector
        'cifar10': (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
        'cifar100': (0.2675, 0.2565, 0.2761),
    }

    cifar_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean[args.in_dataset], cifar_std[args.in_dataset]),
    ]) # meanstd transformation

    cifar_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean[args.in_dataset], cifar_std[args.in_dataset]),
    ])



### STL-10/STL-labeled transforms
transform_train_stl10 = transforms.Compose([
        transforms.RandomCrop(96,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #calculated using snippet above
        transforms.Normalize((0.447, 0.44, 0.4), (0.26, 0.256, 0.271))
    ])

transform_test_stl10=transforms.Compose([
    transforms.ToTensor(),
    #calculated using snippet above
    transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
])

### Tiny Imagenet 200 # copied from STL-10 for now.
transform_train_tin200 = transforms.Compose([
#		transforms.RandomCrop(96,padding=4),
        transforms.Resize(32),
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #calculated using snippet above
        transforms.Normalize((0.447, 0.44, 0.4), (0.26, 0.256, 0.271))
    ])

transform_test_tin200=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    #calculated using snippet above
    transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
])


# #original, mostly untransformed train set.
# transform_train_orig_stl10 = transforms.Compose([
# 	transforms.ToTensor(),
# 	#calculated using snippet above
# 	transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
# ])

#### MNIST Transforms
#from https://github.com/pytorch/examples/blob/master/mnist/main.py
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])


fashion_mnist_train_transform = transforms.Compose([
    transforms.RandomCrop(28,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2868,),(0.3524,))

     ])

fashion_mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,),(0.3530,))

     ])

if(args.in_dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, 
        download=False, transform=cifar_transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False,
        download=False, transform=cifar_transform_test)
    n_class = 10

elif(args.in_dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root=args.datadir, train=True,
        download=False, transform=cifar_transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, 
        download=False, transform=cifar_transform_test)
    n_class = 100

elif(args.in_dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root=args.datadir, train=True, 
        download=True, transform=mnist_transform)
    testset = torchvision.datasets.MNIST(root=args.datadir, train=False, 
        download=True, transform=mnist_transform)
    n_class = 10


elif(args.in_dataset == 'fashion'):
    print("| Preparing Fashion MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.FashionMNIST(root=args.datadir, train=True, 
        download=False, transform=fashion_mnist_train_transform)
    testset = torchvision.datasets.FashionMNIST(root=args.datadir, train=False, 
        download=False, transform=fashion_mnist_test_transform)
    n_class = 10


elif (args.in_dataset == 'stl10-labeled'):
    print("| Preparing STL10-labeled dataset...")
    
    trainset = torchvision.datasets.STL10(root=args.datadir, 
        split='train', download=False, transform=transform_train_stl10)
    testset = torchvision.datasets.STL10(root=args.datadir,
        split='test', download=False, transform=transform_test_stl10)
    n_class = 10

elif (args.in_dataset == 'stl10-c'):
    print("| Preparing STL10-C dataset...")
    import stl10_c
    trainset = stl10_c.STL10_C(root=args.datadir, 
        #split='train', transform=transform_train_stl10, train_list=[[args.train_x,''],[args.train_y,'']])
        split='train', transform=transform_train_stl10, train_list=[[args.train_x,''],[args.train_y,'']])
    testset = stl10_c.STL10_C(root='/home/sunil/ssl/data/stl-10',
        split='test', transform=transform_test_stl10, test_list = [[args.test_x,''],[args.test_y,'']])
    if args.train_y:
        n_class = get_stl10_num_classes(args.train_y)
    else:
        n_class = 10

elif (args.in_dataset == 'tiny-imagenet'):
    #pdb.set_trace()
    import TinyImageNet as tin
    trainset = tin.TinyImageNet(root=args.datadir,split='train',transform=transform_train_tin200,in_memory=False, download=False)
    testset = tin.TinyImageNet(root=args.datadir,split='val',transform=transform_test_tin200,in_memory=False, download=False)
    n_class = 200

else:
    print("Unknown data set")
    sys.exit(0)

sys.stdout.flush()


train_in_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_in_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)


### Load OoD Dataset if it has been specified
if args.out_dataset is None:
    ood_trainloader = None

else:
    if (args.out_dataset == "tiny-images"):
        # pdb.set_trace()
        print("loading Tiny Images")        
        import tiny_image_data 

        x = np.load(args.ood_train_x)
        #labels for OoD dataset will be assigned to class K. (In-distribution labels go from 0 to K-1)
        y = n_class * np.ones(x.shape[0],dtype=np.int64)
        # print("| Preparing TinyImage/CIFAR-100 dataset...")

        if args.in_dataset == 'cifar10' or args.in_dataset == 'cifar100':
            train_transform = cifar_transform_train
            test_transform = cifar_transform_test
        # elif args.dataset == 'stl10-c':
        # 	train_transform = transform_train_stl10
        else:
            print("Unsupported in-distribution/out-of-distribution pairing")
            sys.exit(0)

        train_out_dataset = tiny_image_data.TinyImage(x, y, train=True, transform=train_transform)
        val_out_dataset = tiny_image_data.TinyImage(x, y, train=False, transform=test_transform)

        val_out_loader =  torch.utils.data.DataLoader(val_out_dataset, batch_size = math.ceil(args.batch_size/n_class), shuffle=True, num_workers=1)
        
        

    elif (args.out_dataset == 'stl10-unlabeled'):
        #pdb.set_trace()
        print("loading STL-10 unlabeled data")
        train_out_dataset = stl10_c.STL10_C(root=args.datadir, 
                split='unlabeled',transform=transform_train_stl10)
        #the PyTorch STL_10 loader does assigns labels of -1 to the unlabeled split. So change that here.
        train_out_dataset.labels = n_class*np.ones(train_out_dataset.data.shape[0],dtype=np.int64)

    else:
        print("Unsupported OoD Dataset")
        sys.exit(0)
    


use_cuda=True
# if args.use_gpu:
# #if use_cuda:
# 	cuda_device = None
# 	while(cuda_device is None):
# 		cuda_device = gpu_utils.get_cuda_device(args)
# 		use_cuda = True

# if use_cuda:
#     torch.cuda.manual_seed(args.seed)


#the default network to use if no network is specified at command line.
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=args.dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        #pdb.set_trace()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# get available gpus
if args.gpus[0] < 0:
    import GPUtil
    n_gpus = -args.gpus[0] if args.gpus[0] < -1 else 4
    args.gpus = [int(i) for i in GPUtil.getAvailable(order='first', limit=n_gpus, maxMemory=0.15)]
    if len(args.gpus) < n_gpus:
        raise RuntimeError('No enough GPUs')
    print('Using GPUs:', *args.gpus)
torch.cuda.set_device(args.gpus[0])
print("Number of classes:",n_class)

model = ConvNet(n_class)

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