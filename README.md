# NAS(Neural Architecture Search)

## FBNet 
- Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**Implementation of FBNet with MXNet**

paper address: https://arxiv.org/pdf/1812.03443.pdf

### Implemented Net:

- FBNet
- FBNet Based on Se_Resnet_50_Architecture

other block_type architecture cound be easily implement by modify fbnet-symbol/block.py

**Code:**
* `blocks.py`: Define blocks symbols
* `FBNet.py`: Define FBNet Class.
* `FBNet_SE.py`: Define FBNet Architecture  based on Se_resnet_50.
* `blocks_se.py`: Define blocks symbols based on new search space,include [Resnet_50,Se,Group_Conv,Channel_shuffle,Deform_Conv]
* `util.py`: Define some functions.
* `test.py`: Run test.
* `block_speed_test.py`: test block lat in real environment(1080Ti)


**Differences from original paper**: 
  * The last conv layer's num_filters is repalced by feature_dim specified by paramters
  * Use *Amsoftmax*, *Arcface* instead of *FC*, but you can set model_type to `softamx` to use fc
  * Default input shape is `3,108,108`, so the first conv layer has stride 1 instead of 2.
  * Add `BN` out of blocks, and **no** `bn` inside blocks.
  * Last conv has kernel size `3,3`
  * Use **+** in loss not **\***.
  * Adding gradient rising stage in cosine decaying schedule. Code in fbnet-symbom/util/CosineDecayScheduler_Grad
  

#### How to train:

If you want to modify the network structure or the learning rate adjustment function, you need to modify the source code, 
otherwise you can use this command directly:

```shell
python test.py --gpu 0,1,2,3,4,5,6  --log-frequence 50 --model-type softmax --batch-size 32 
```

#### How to retrain:

When we want to train the large dataset and hope to change learning rate manually, or the machine is suddenly shutdown due to some reason,
of course, we definitely hope we can continue to train model with previous trained weights. Then, your can use this cmd:

```shell
python test.py --gpu 0,1,2,3,4,5,6  --log-frequence 50 --model-type softmax --batch-size 32 --load-model-path ./model
```
This can load the latest model params for retrain,If you want to load the model with specific epoch,
you can use ** --load-model-path ./model/*.params **,This means you can retrain your model from specific model.


*TODO*:
  - sample script, for now just save $\theta$
  - ~~cosine decaying schedule~~
  - ~~lat in real environment~~
  - ~~DataParallel implementation~~