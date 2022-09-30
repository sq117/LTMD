# LTMD
This is the PyTorch implementation of paper: LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout.

Here we provide code for CIFAR10 dataset as a pre-release version. More datasets will be added later.

## Dependencies and Installation
- Python 3.8.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- PyThorch 1.9.0
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Dataset Preparation
As for MNIST and CIFAR10, the data can be downloaded by torchvision as in the code. The DVS-CIFAR10 is avaliable at [here](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2). The N-MNIST dataset is avaliable at [here](https://www.garrickorchard.com/datasets/n-mnist).

Preprocessing of splitting images of N-MNIST and DVS-CIFAR10 are provided as [dataset.py](https://github.com/sq117/LTMT/blob/main/nmnist/dataset.py) and [processing_data.py](https://github.com/sq117/LTMT/blob/main/dvscifar10/processing_data.py) in the corresponding folders.

## Training and Testing
In the corresponding directory, run following command with GPU IDs identification:

	CUDA_VISIBLE_DEVICE=0,1,2,3 python trialnew.py

Network will be trained and tested iteratively. Highest performance model and the final model generated in the last epoch will be automatically saved in tmp folder.

You can also find our trained models in tmp folder.

All the hyperparameters are initialized same as the values mentioned in the paper.

## Load and Test the Existing Model
In the corresponding directory, run following command with GPU IDs identification:

	CUDA_VISIBLE_DEVICE=0,1,2,3 python test.py

Model with the highest accuracy will be loaded and run testing process.

## Results
The results of our LTMD method on MNIST, N-MNIST, CIFAR10 and DVS-CIFAR10 are:

|    Model    | Time steps |  Best  |  Mean   |   Std   |
|:-----------:|:----------:|:------:|:-------:|:-------:|
|    MNIST    |     4      | 99.60% | 99.584% | 0.012%  |
|   CIFAR10   |     4      | 94.19% | 94.154% | 0.0403% |
|   N-MNIST   |     15     | 99.65% | 99.614% | 0.0206% |
| DVS-CIFAR10 |     7      | 73.30% | 72.92%  | 0.319%  |

---

Results of ablation study on the effect of our proposed methods are:

|    Model    | Time steps | LIF (plain) | LT (Learnable thresholding) | LTMD (Learnable threshodling&Moderate dropout) |
|:-----------:|:----------:|:-----------:|:-------------------------:|:--------------------------------------------:|
|    MNIST    |     4      |   99.53%    |          99.57%           |                    99.60%                    |
|   CIFAR10   |     2      |   92.88%    |          93.51%           |                    93.75%                    |
|   N-MNIST   |     15     |   99.55%    |          99.58%           |                    99.65%                    |
| DVS-CIFAR10 |     7      |   71.30%    |          72.30%           |                    73.30%                    |

---

Performance evaluation of LTMD with DenseNet and VGG architectures:

| SNN architecture |  LIF   |   LT   |  LTMD  |
|:----------------:|:------:|:------:|:------:|
|     ResNet19     | 91.97% | 93.59% | 93.78% |
|      VGG16       | 91.68% | 92.10% | 92.35% |

---

The computational load and learnable parameters of SNNs. Percentage increased based on static thresholding is shown in brackets:

|  Computational load  | Static threshold |         LT          |        LTMD         |  ANN  |
|:--------------------:|:----------------:|:-------------------:|:-------------------:|:-----:|
|    Training time     |    5762.92min    | 5955.76min (+3.35%) | 6086.49min (+5.61%) |       |
| Learnable parameters |      6.82M       |  6.82M (+0.0015%)   |  6.82M (+0.0015%)   |       |
|      #Additions      |     196.83M      |  214.81M (+9.14%)   |  212.55M (+7.99%)   | 648.46M |
|   #Multiplications   |      28.85M      |     28.85M (0%)     |  28.86M (+0.035%)   | 648.46M |

---

SNN performance under different noise levels:

| Model  | Noise=0 | Noise=0.1 | Noise=0.2 | Noise=0.3 | Noise=0.4 | Noise=0.5 |
|:------:|:-------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|  SNN   | 99.53%  |  90.60%   |  82.33%   |  75.76%   |  54.16%   |  31.56%   |
| SNN&LI | 99.60%  |  96.54%   |  92.10%   |  83.61%   |  69.05%   |  35.79%   |
