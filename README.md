## Updates
- July 12, 2026: Optimized pytorch CUDA implementation.
- November 2, 2023: Added support for Mixed Precision.
- March 14, 2023: Added support for PyTorch (latest for pytorch 2.1.0).

## Code-for-MPELU
Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units, [arXiv:1606.00305](https://arxiv.org/abs/1606.00305)

The main contributions are:

- A new activation function, MPELU, which is a unified form of ReLU, PReLU and ELU.
- A weight initialization method for both ReLU-like and ELU-like networks. If used with the ReLU nework, it reduces to Kaiming initialization.
- A network architecture that is more effective than the original Pre-/ResNet.

#### Citation
```
@article{LI201811,
		title = "Improving deep neural network with Multiple Parametric Exponential Linear Units",
		journal = "Neurocomputing",
		volume = "301",
		pages = "11 - 24",
		year = "2018",
		issn = "0925-2312",
		doi = "https://doi.org/10.1016/j.neucom.2018.01.084",
		author = "Yang Li and Chunxiao Fan and Yong Li and Qiong Wu and Yue Ming"
}
```

## Deep MPELU residual architecture

MPELU nopre bottleneck architecture:

![img](torch/models/MPELU-NoPre-ResNet.jpg)

## Experiments on CIFAR-10/100

MPELU is initialized with alpha = 0.25 or 1 and beta = 1. The learning rate multipliers of alpha and beta are 5. The weight decay multipliers of alpha and beta are 5 or 10. The results are reported as best(mean ± std).

MPELU nopre ResNet | depth | #params | CIFAR-10 | CIFAR-100
-------|:--------:|:--------:|:--------:|:--------:|
alpha = 1; beta = 1 | 164 | 1.696M | 4.58 (4.67 ± 0.06) | 21.35 (21.78 ± 0.33)
alpha = 1; beta = 1 | 1001 | 10.28M | 3.63 (3.78 ± 0.09) | 18.96 (19.08 ± 0.16)
alpha = 0.25; beta = 1 | 164 | 1.696M | 4.43 (4.53 ± 0.12) | 21.69 (21.88 ± 0.19)
alpha = 0.25; beta = 1 | 1001 | 10.28M | **3.57 (3.71 ± 0.11)** | **18.81 (18.98 ± 0.19)**

The experimental results in paper were conducted in torch7. But we also provide `pytorch` and `caffe` implementations. If you want to use the torch7 version to replicate our results, please follow the steps below: 

1. Install [fb.resnet.troch](https://github.com/facebook/fb.resnet.torch)
2. Follow our instructions to install MPELU in torch.
2. Copy files in `mpelu_nopre_resnet` to `fb.resnet.torch` and overwrite the original files.
3. Run the following command to train a 1001-layer MPELU nopre ResNet

```
th main.lua -netType mpelu-preactivation-nopre -depth 1001 -batchSize 64 -nGPU 2 -nThreads 12 -dataset cifar10 -nEpochs 300 -shortcutType B -shareGradInput false -optnet true | tee checkpoints/log.txt
```

## Installation
We now provide [PyTorch](https://pytorch.org/), [Caffe](https://github.com/BVLC/caffe) and [Torch7](http://torch.ch/)(deprecated) implementations.

### PyTorch

The pytorch version is implemented using CUDA for fast computation. The code has been tested in Ubuntu 20.04 with CUDA 11.6. The implementation is isolated from your PyTorch library and does not modify any other Python packages installed on your system. It can be installed and uninstalled independently using the `pip` package manager, and therefore can be used alongside your original PyTorch library without interfering with its functionality. You may integrate them into your projects as needed.

1) `cd ./pytorch`

2) `pip install .`

### Caffe:

1) Download the latest `caffe` from [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)

2) Move `caffe/*` of this repo to the `caffe` directory and follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to compile.

### Torch7:

1) Update `torch` to the latest version. This is necessary because of [#346](https://github.com/torch/cunn/pull/346).

2) Move `torch/extra` in this repo to the official torch directory and overwrite the corresponding files.

3) Run the following command to compile new layers.

```
cd torch/extra/nn/
luarocks make rocks/nn-scm-1.rockspec
cd torch/extra/cunn/
luarocks make rocks/cunn-scm-1.rockspec
```

## Usage
### PyTorch

Examples:
```
# install MPELU first, then
python examples/mnist_mpelu.py
```

To use the MPELU module in a neural network, you can import it from the mpelu module and then use it as a regular PyTorch module in your network definition.

For example, let's say you have defined the MPELU module in a file called mpelu.py. To use it in a neural network, you can do the following:

```
import torch
from mpelu import MPELU

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.mpelu1 = MPELU(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.mpelu2 = MPELU(32)
        self.fc = torch.nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mpelu1(x)
        x = self.conv2(x)
        x = self.mpelu2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x
```

### Caffe:

**MPELU**:
In caffe, MPELU exists as the `M2PELU` layer, where `2` for `two parameters` alpha and beta which are both initialized as 1 in default.
To simply use this layer, replace `type: "ReLU"` with `type: "M2PELU"` in network defination files.

**Taylor filler**:
First, replace the keyword `gaussian` or `MSRA` with `taylor` in the `weight_filler` domain. Then, Add two new lines to specify values of `alpha` and `beta`:

```
weight_filler {
      type: "taylor"
      alpha: 1
      beta: 1
}
```
See the examples for details.


### Torch7

I implemented two activation functions, `SPELU` and `MPELU`, where `SPELU` is a trimmed version of MPELU and can also be seen as a learnable `ELU`.

```
nn.SPELU(alpha=1, nOutputPlane=0)
nn.MPELU(alpha=1, beta=1, nOutputPlane=0)
```

- When `nOutputPlane = 0`, the `channel-shared` version will be used. 
- When `nOutputPlane` is set to the number of feature maps, the `channel-wise` version will be used.

To set the multipliers of weight decay for `MPELU`, use the `nnlr` package.

```
$ luarocks install nnlr
```

```
require 'nnlr'

nn.MPELU(alpha, beta, channels):learningRate('weight', lr_alpha):weightDecay('weight', wd_alpha)
                               :learningRate('bias', lr_beta):weightDecay('bias', wd_beta)
```

**Taylor filler**: Please check our examples in `mpelu_nopre_resnet`.
