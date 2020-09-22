import  torch
import torch.nn.functional as F
import numpy as np
import argparse
from datasets.tiny_image_data import TinyImage

parser = argparse.ArgumentParser()

'''scripts
python save_tinyimages_subset.py -i /home/zeus/data/tiny_images/tiny_images_rand_100k.npy -o tiny_images_rand_100k.npy -r random
python save_tinyimages_subset.py -i tiny_images_rand_100k.npy -o tiny_images_rand_10k_adversarial_resample.npy -r checkpoints/cifar10_tiny_images_weights.pth
'''

parser.add_argument('-i','--input-path', type=str, default='/home/zeus/data/tiny_images/tiny_images_rand_100k.npy')
parser.add_argument('-o','--output-path', type=str, default='tiny_images_rand_10k_adversarialResample.npy')
parser.add_argument('-p','--ratio', type=str, default=0.1)
parser.add_argument('-r', '--resample', type=str, default='random')

args = parser.parse_args()

x = np.load(args.input_path)
train_out_dataset = TinyImage(x, x, train=True, transform=None)

if args.resample == 'random':
    w = torch.rand(len(train_out_dataset))
    w_thresh = torch.kthvalue(w, int((1 - args.ratio) * len(w)))[0]
    keep_idx = [c.item() for c in (w >= w_thresh).nonzero()]
else:
    w = torch.load(args.resample, map_location=lambda storage, loc: storage).detach()
    p = F.softplus(w)
    p = args.ratio * p / p.mean()
    keep_idx = [c.item() for c in (p >= torch.rand(len(w))).nonzero()]

subset = x.take(keep_idx, axis=0)

np.save(args.output_path, subset)

_subset = np.load(args.output_path)
print(x.shape, subset.shape, _subset.shape)
np.testing.assert_array_equal(subset, _subset)