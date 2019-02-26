# SNAS
[SNAS: STOCHASTICS NEURAL ARCHITECTURE SEARCH](https://arxiv.org/abs/1812.09926)

**Reference code:**
- https://github.com/Astrodyn94/SNAS-Stochastic-Neural-Architecture-Search-
- https://github.com/quark0/darts


**Train cifar10**

```shell
python train_cifar10.py --batch-size 64 --gpus 0,1 --log-frequence 20
```

**Train face**

Before training, you should link data preprocessing code from fbnet-pytorch.
Run following code under `${nas_root_path}/snas/snas/` directory.

```shell
ln -s ../../fbnet-pytorch/data.py data.py
ln -s ../../fbnet-pytorch/data_face.py data_face.py
ln -s ../../fbnet-pytorch/tmp tmp
```

Training:

```shell
python train_face.py --batch-size 64 --gpus 0,1 --log-frequence 20
```

**Features:**

- Support multi-gpus training
