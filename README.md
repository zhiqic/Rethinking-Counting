# **Rethinking Spatial Invariance of Convolutional Networks for Object Counting**

## Highlights

- 🏆 **2022 Pulitzer Prize for Public Service**: Application in analyzing the Capitol Riot earned a [Pulitzer Prize for public service](https://www.pulitzer.org/prize-winners-by-year). 
- 🎬 Watch the [YouTube Demonstration](https://www.youtube.com/watch?v=WiZ51V5M0C8&ab_channel=Zhi-QiCheng).
- 📰 Read about it on the [Washington Post](https://www.washingtonpost.com/investigations/interactive/2021/dc-police-records-capitol-riot/).

In this project, we explore a unique approach to convolutional networks and object counting. This repository contains the full implementation of our paper, [Rethinking Spatial Invariance of Convolutional Networks for Object Counting](https://arxiv.org/pdf/2206.05253.pdf). Here, we introduce the GauNet implementation in C++ and CUDA, supplemented by a TensorFlow plugin. Dive into our innovative method of using locally connected Gaussian kernels to elevate the process of object counting in convolutional networks. Developed with support from Vitjan Zavrtanik (VitjanZ). Developed with support from Vitjan Zavrtanik (VitjanZ). Many implementations of this version are from [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [DAU-ConvNet](https://github.com/skokec/DAU-ConvNet).


<p align="center">
  <img width="1170" alt="image" src="https://github.com/zhiqic/Rethinking-Counting/assets/65300431/2aea7e32-5d7e-4514-a321-8fbb2facb6ea">
</p>


## Quick Start

### TensorFlow Plugin

Easily replace the `tf.contrib.layers.conv2d` function using our TensorFlow plugin and Python wrappers.

**Dependencies**:
- Python2.7 or Python3.5
- TensorFlow 1.6+
- Numpy
- OpenBlas

Optional:
- Scipy
- Matplotlib
- Python-tk for unit tests

```bash
# Install the required dependencies
sudo apt-get install python2.7 python3.5 python-pip python3-pip
pip install tensorflow==1.6 numpy openblas
```

### Installation

For a quick setup, use the pre-compiled binaries for `TensorFlow`:

```bash
# Install further dependencies
sudo apt-get install libopenblas-dev wget

# Set up the package
export TF_VERSION=1.13.1
sudo pip install link_to_dau_conv_package
```

For Docker users, access our pre-built images on [Docker Hub](https://hub.docker.com/r/skokec/dau-convnet). To manually build and install, please refer to the detailed steps in the original [DAU-ConvNet](https://github.com/skokec/DAU-ConvNet).

### Training & Testing

Training:

1. Set the parameters in `config.py` and `./datasets/XXX/setting.py`. If you wish to reproduce our results, we recommend using our parameters in `./results_reports`.
2. Run the command: `python train.py`.
3. Monitor the training with TensorBoard using: `tensorboard --logdir=exp --port=6006`.

Testing:

- We only provide an example to test the model on the test set. You may need to modify it to test your own models.

> **Note**: The training and testing instructions align with the ones used in the [C^3 Framework](https://github.com/gjy3035/C-3-Framework). You can follow similar commands as specified in that framework.


## Acknowledgments 

Our heartfelt appreciation goes to:

- **Vitjan Zavrtanik (VitjanZ)** for his pivotal support in developing the TensorFlow version of our project.
  
- The teams behind [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [DAU-ConvNet](https://github.com/skokec/DAU-ConvNet). Their work has significantly influenced our TensorFlow implementation.


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

> **Disclaimer**: Due to the presence of sensitive information, modifications were made to this repository. Unfortunately, the PyTorch version is not available at this time. Thank you for your understanding.
