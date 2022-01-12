# HyperTune

HyperTune is a fully distributed hyperparameter optimization tool for PyTorch DNNs.
Distribute your hyperparameter trials across remote machines, and select from a
variety of parallel DNN training strategies to distribute training across 
available GPUs.

## Installation
First, install the required dependencies into a virtual environment.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run *ImageNet* experiments, you must install and preprocess the ImageNet dataset.

1. Download the ImageNet dataset from [Kaggle](
https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz
). We recommend using the Kaggle API to do this, since the file is very large.
2. Fully unzip the downloaded file
3. Copy and run the [valprep.sh](
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
) script to move the validation images to labelled subfolders.
4. Once completed, you will need the full path to run our script. It should look
something like `/johndoe/datasets/ILSVRC/Data/CLS-LOC`.

## Running HyperTune
To run HyperTune, use the `run_hypertune.sh` script. This script provides a generic
runner that can execute any DNN training script that prints the expected output.
We have provided examples for two datasets / tasks (ImageNet and MNIST)
and two DNN models (ResNet and AlexNet).

> Note: `run_hypertune.sh` hardcodes the expectation of 3 remote machines, aliased as
gpu1, gpu2, and gpu3. For our experiments, we also hardcode 1 epoch, and a few minor
arguments. To change these, simply edit the script before running.

When prompted by the script, provide the following paths in addition to the other
parameters.

### MNIST:

| File                        | Path Within Repo                                     |
|-----------------------------|------------------------------------------------------|
| Training File               | ./models/MNIST/train.py       |
| Hyperparameter Space Config | ./models/MNIST/hyperparameter_space_MNIST.json |

### Imagenet:

| File                        | Path Within Repo                                     |
|-----------------------------|------------------------------------------------------|
| Training File               | ./models/ImageNet/train.py                           |
| Hyperparameter Space Config | ./models/ImageNet/hyperparameter_space_ImageNet.json |


## Running Horovod + Ray Tune
To evaluate HyperTune, we compare against the popular Ray Tune tool backed by Horovod.
To run this benchmark, use the `run_horovod_raytune.sh` script. This script starts a Ray
cluster on your local machine, so run it on whichever machine you intend to be your Ray
**head node**.

> Note: `run_horovod_raytune.sh` hardcodes the specification of 1 epoch, and a few minor
arguments. `ray_cluster.yaml` hardcodes the IP addresses of the head and worker nodes,
along with SSH username for logging in to worker nodes. To change these, simply edit the
script before running.

## Notes
Due to time constraints, the included AlexNet model is not compatible with GPipe. The
`torchgpipe` library (which is used to provide GPipe support) requires that all PyTorch
models inherit from `nn.Sequential`. Therefore, adding support for AlexNet (or any other
non-sequential DNN you wish to support) requires a custom `nn.Sequential` implementation.
Please refer to the `torchgpipe` documentation for more information.

## Results
For more information about this project and our findings, please see our paper, located
within this repo at `results/HyperTune.pdf`.
