# [CVPR 2022] Rethinking Spatial Invariance of Convolutional Networks for Object Counting


## 2022 Pulitzer Prize for public service
Our method was applied to the analysis of the Capitol Riot by the Washington Post and won the [**2022 Pulitzer Prize for public service**](https://www.pulitzer.org/prize-winners-by-year). You can watch the video demonstration on [YouTube](https://www.youtube.com/watch?v=WiZ51V5M0C8&ab_channel=Zhi-QiCheng) and read the news reports on the [Washington Post](https://www.washingtonpost.com/investigations/interactive/2021/dc-police-records-capitol-riot/).

<p align="center">
  <img src="./figures/demo.gif" alt="animated" />
</p>


## Overview
This repository contains the implementation of the paper [**Rethinking Spatial Invariance of Convolutional Networks for Object Counting**](https://arxiv.org/pdf/2206.05253.pdf). It includes a self-contained GauNet implementation in C++ and CUDA, as well as a TensorFlow plugin. The library implements DAU layers for any deep learning framework.

We propose a low-rank approximation with translation invariance to efficiently approximate massive Gaussian convolution. Our method uses locally connected Gaussian kernels to replace the original convolution filters, which helps estimate the spatial position in the density map. This allows the feature extraction process to potentially stimulate the density map generation process, thus overcoming annotation noise. Our work provides a new direction for future research to investigate how to properly relax the strict pixel-level spatial invariance for object counting.

![framework](./figures/framework.png)


## Available implementations
We are grateful to Vitjan Zavrtanik (VitjanZ) for TensorFlow C++/Python wrapper.
The training script borrows some codes from the [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [DAU-ConvNet](https://github.com/skokec/DAU-ConvNet) repositories. Currently, there are some inconsistencies in the TensorFlow version, so we recommend waiting for our PyTorch implementation.
- [x] TensorFlow version
- [-] PyTorch vsrsion 

See below for more details on each implementation.


## TensorFlow
We provide a TensorFlow plugin and accompanying Python wrappers that can be used to directly replace the `tf.contrib.layers.conv2d` function. Our C++/CUDA code only supports NCHW format for input, so you'll need to update your TensorFlow models accordingly.


Requirements and dependency libraries for TensorFlow plugin:
 * Python (tested on Python2.7 and Python3.5)
 * TensorFlow 1.6 or newer 
 * Numpy
 * OpenBlas
 * (optional) Scipy, matplotlib and python-tk  for running unit test in `dau_conv_test.py`
 
## Installation from pre-compiled binaries (pip)
If you are using `TensorFlow` from pip, then install pre-compiled binaries (.whl) from the [RELEASE](https://github.com/skokec/DAU-ConvNet/releases) page (mirror server also available http://box.vicos.si/skokec/dau-convnet):

```bash
# install dependency library (OpenBLAS)
sudo apt-get install libopenblas-dev  wget

# install dau-conv package
export TF_VERSION=1.13.1
sudo pip install https://github.com/skokec/DAU-ConvNet/releases/download/v1.0/dau_conv-1.0_TF[TF_VERSION]-cp35-cp35m-manylinux1_x86_64.whl
```

Note that pip packages were compiled against the specific version of TensorFlow from pip, which must be installed beforehand.

## Docker 
Pre-compiled docker images for TensorFlow are also available on [Docker Hub](https://hub.docker.com/r/skokec/dau-convnet) that are build using the [`plugins/tensorflow/docker/Dockerfile`](https://github.com/skokec/DAU-ConvNet/blob/master/plugins/tensorflow/docker/Dockerfile). 

Dockers are build for specific python and TensorFlow version. Start docker, for instance, for Python3.5 and TensorFlow r1.13.1, using:

```bash
sudo nvidia-docker run -i -d -t skokec/tf-dau-convnet:1.0-py3.5-tf1.13.1 /bin/bash
```

## Build and installation ##
Requirements and dependency libraries to compile DAU-ConvNet:
 * Ubuntu 16.04 (not tested on other OS and other versions)
 * C++11
 * CMake 2.8 or newer (tested on version 3.5)
 * CUDA SDK Toolkit (tested on version 8.0 and 9.0)
 * BLAS (ATLAS or OpenBLAS)
 * cuBlas

On Ubuntu 16.04 with pre-installed CUDA and cuBLAS (e.g. using nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 or nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 docker) install dependencies first:

```bash
apt-get update
apt-get install cmake python python-pip libopenblas-dev
 
pip install tensorflow-gpu>=1.6
# Note: during installation TensorFlow package is sufficient, but during running, the TensorFlow-GPU is required.
```

Then clone the repository and build from the source:
```bash
git clone https://github.com/skokec/DAU-ConvNet
git submodule update --init --recursive

mkdir DAU-ConvNet/build
cd DAU-ConvNet/build

cmake -DBLAS=Open -DBUILD_TENSORFLOW_PLUGIN=on ..

make -j # creates whl file in build/plugin/TensorFlow/wheelhouse
make install # will install whl package (with .so files) into python dist-packages folder 

```

## Preparation 
- Clone this repo in the directory (```Root/GauNet```):
- Install dependencies. We use python 3.7 and PyTorch >= 1.6.0 : http://pytorch.org.

    ```bash
    conda create -n GauNet python=3.7
    conda activate GauNet
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
    cd GauNet
    ```

## Training
Check the parameters in ```config.py``` before training.
Please refer to [C^3 Framework](https://github.com/gjy3035/C-3-Framework) for more details.

## Testing
Please refer to [C^3 Framework](https://github.com/gjy3035/C-3-Framework) for more details.


## Citation
Please cite our CVPR 2022 paper:
```
@InProceedings{Cheng_2022_CVPR,
    author    = {Cheng, Zhi-Qi and Dai, Qi and Li, Hong and Song, Jingkuan and Wu, Xiao and Hauptmann, Alexander G.},
    title     = {Rethinking Spatial Invariance of Convolutional Networks for Object Counting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month     = {June},
    year      = {2022},
    pages     = {19638-19648}
}
```

## Acknowledgment
This work was partially supported by the Air Force Research Laboratory under agreement number FA8750-19-2-0200, the financial assistance award 60NANB17D156 from the U.S. Department of Commerce, National Institute of Standards and Technology (NIST), the Intelligence Advanced Research Projects Activity (IARPA) via the Department of Interior/Interior Business Center (DOI/IBC) contract number D17PC00340, and the Defense Advanced Research Projects Agency (DARPA) grant funded under the GAILA program (award HR00111990063).

The U.S. government is authorized to reproduce and distribute reprints for governmental purposes, notwithstanding any copyright notation. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the Air Force Research Laboratory or the U.S. government.

## License
GauNet is released under the Apache 2.0 license.
