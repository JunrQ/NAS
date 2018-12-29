# FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**MXNet** implementation.

**To run with mnist**: Open an terminal, and run the code
```shell
python run_mnist.py --batch-size 32 --gpu 1 --log-frequence 50
```

**Code:**
* `blocks.py`: Define blocks symbols
* `FBNet.py`: Define FBNet Class.
* `util.py`: Define some functions.
* `test.py`: Run test.

**Differences from original paper**: 
  * The last conv layer's num_filters is repalced by feature_dim specified by paramters
  * Use *Amsoftmax*, *Arcface* instead of *FC*, but you can set model_type to `softamx` to use fc
  * Default input shape is `3,108,108`, so the first conv layer has stride 1 instead of 2.
  * Add `BN` out of blocks, and **no** `bn` inside blocks.
  * Last conv has kernel size `3,3`
  * Use **+** in loss not **\***.

*TODO*:
  - sample script, for now just save $\theta$
  - ~~cosine decaying schedule~~
  - lat in real environment
  - ~~DataParallel implementation~~