# FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**MXNet** implementation.

* `blocks.py`: Define blocks symbols
* `FBNet.py`: Define FBNet Class.
* `util.py`: Define some functions.
* `test.py`: Run test.

**Differences from original paper**: 
  * The last conv layer's num_filters is repalced by feature_dim specified by paramters
  * Use *Amsoftmax*, *Arcface* instead of *FC*, but you can set model_type to `softamx` to use fc
  