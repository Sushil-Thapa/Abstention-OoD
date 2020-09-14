### For Sushil's Personal Use
```
python train.py --gpus 0 -a resnet18 -d cifar10 --epochs 10 --save-freq 10 --print-freq 40 2>&1 | tee checkpoints/train_vanilla_dac.log
```

```
python cv_train_bg_resample.py --gpus 0 -a resnet18 -id cifar10 -od tiny_images --epochs 10 --save-freq 10 --print-freq 40 2>&1 | tee checkpoints/train_bg_resample_dac.log
```

```
python train_bg.py --gpus 0 -a resnet18 -id cifar10 -od tiny_images   --epochs 5 --save-freq 5 --print-freq 40 --load-path checkpoints/cifar10_resnet18_10ep-7977top1.pth --resample checkpoints/cifar10_tiny_images_resnet18_resample_10ep-weights.pth  -p 0.1  --lr 0.01 2>&1 | tee checkpoints/train_bg_finetune_resample.log
```

