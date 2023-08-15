# **Rethinking Spatial Invariance of Convolutional Networks for Object Counting**
> **CVPR 2022** 

## Highlights

- 🏆 **2022 Pulitzer Prize for Public Service**: Our method's application in analyzing the Capitol Riot earned a [Pulitzer Prize for public service](https://www.pulitzer.org/prize-winners-by-year). 
- 🎬 Watch the [YouTube Demonstration](https://www.youtube.com/watch?v=WiZ51V5M0C8&ab_channel=Zhi-QiCheng).
- 📰 Read about it on the [Washington Post](https://www.washingtonpost.com/investigations/interactive/2021/dc-police-records-capitol-riot/).

## Introduction

In this project, we explore a unique approach to convolutional networks and object counting. This repository contains the full implementation of our paper, [Rethinking Spatial Invariance of Convolutional Networks for Object Counting](https://arxiv.org/pdf/2206.05253.pdf). Here, we introduce the GauNet implementation in C++ and CUDA, supplemented by a TensorFlow plugin. Dive into our innovative method of using locally connected Gaussian kernels to elevate the process of object counting in convolutional networks.

<p align="center">
  <img width="683" alt="image" src="https://github.com/zhiqic/Rethinking-Counting/assets/65300431/d077c925-42a7-4d6b-b0f7-247dc27fc530">
</p>

## Implementations

- **TensorFlow Version** (Current): Developed with support from Vitjan Zavrtanik (VitjanZ). This version contains references from [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [DAU-ConvNet](https://github.com/skokec/DAU-ConvNet). Please note some inconsistencies are currently being addressed.
- **PyTorch Version**: Unfortunately, due to various constraints, we are unable to provide a PyTorch version at this time.

## Quick Start

### TensorFlow Plugin

Easily replace the `tf.contrib.layers.conv2d` function using our TensorFlow plugin and Python wrappers.

**Dependencies**:
- Python2.7 or Python3.5
- TensorFlow 1.6+
- Numpy
- OpenBlas
- Optional: Scipy, Matplotlib, Python-tk for unit tests

### Installation

For a quick setup, use the pre-compiled binaries for `TensorFlow`:

```bash
# Install dependencies
sudo apt-get install libopenblas-dev wget

# Set up the package
export TF_VERSION=1.13.1
sudo pip install link_to_dau_conv_package
```

For Docker users, access our pre-built images on [Docker Hub](https://hub.docker.com/r/skokec/dau-convnet). To manually build and install, please refer to the detailed steps in the original content.

### Training & Testing

Configure your training parameters using `config.py`. For extensive details on training and testing, consult the [C^3 Framework](https://github.com/gjy3035/C-3-Framework).

## Citing Our Work

If you use our work in your research, kindly cite our paper:

```bibtex
@InProceedings{Cheng_2022_CVPR,
    author    = {Cheng, Zhi-Qi and Dai, Qi and Li, Hong and Song, Jingkuan and Wu, Xiao and Hauptmann, Alexander G.},
    title     = {Rethinking Spatial Invariance of Convolutional Networks for Object Counting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month     = {June},
    year      = {2022},
    pages     = {19638-19648}
}
```

## License

GauNet is licensed under the [Apache 2.0 license](LICENSE.md).

> **Note**: This repository is solely for academic purposes. Some information cannot be uploaded to GitHub due to restrictions. We appreciate your understanding.
