# NAS(Neural Architecture Search)

## FBNet 
- Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**Implementation of [FBNet](https://arxiv.org/pdf/1812.03443.pdf) with PyTorch**

## Train cifar10
```shell
python train_cifar10.py --batch-size 32 --log-frequence 100
```

## Train ImageNet
Randomly choose 100 classes from 1000.
You need specify the root dir `imagenet_root` of ImageNet in `train.py`.
```shell
python train.py --batch-size 32 --log-frequence 100
```
