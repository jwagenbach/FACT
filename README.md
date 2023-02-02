# Reproducing "Label-Free Explainability for Unsupervised Models"

This repository is a reproduction of the [Label-Free Explainability for Unsupervised Models](https://arxiv.org/abs/2203.01928). It is heavily based on the authors originally code, which is available [here](https://github.com/JonathanCrabbe/Label-Free-XAI).

## Setup

We've saved the conda environment we used, so it can be reinstalled with
```
conda env create --file environment.yml
```

## Experiments

### (Claim 1) Consistency checks

Validating if both feature and example importance scores are valid and measures what it meant to.

#### (Claim 1.1) Feature importance

Checking if feature selection based on importance scores are consistent, meaning if we mask out the most important pixels, we cause highest representation shift than if we masked out less important pixels. 

```
# MNIST
python experiments/mnist.py --name consistency_features

# CIFAR10
python experiments/cifar100_from_mnist.py --name consistency_features --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name consistency_features --data cifar100
```

#### (Claim 1.2) Example importance 

Checking if example selection based on importance scores are consistent, meaning the train examples with higher scores share similar labels.

```
# MNIST
python experiments/mnist.py --name consistency_examples

# CIFAR10
python experiments/cifar100_from_mnist.py --name consistency_examples --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name consistency_examples --data cifar100
```

### (Claim 2) Correlation

Checking the Pearson correlation of both of these importance scores. This script outputs a latex table file, in which the first table refers to the feature importance Pearson correlation, and the third one is the same for example importance.

```
# MNIST
python experiments/mnist.py --name pretext

# CIFAR10
python experiments/cifar100_from_mnist.py --name pretext --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name pretext --data cifar100
```

### (Claim 3) Disentanglement

Checking the Pearson correlation of the most important pixels between VAE networks that were trained for different tasks plus a classification network.

```
# MNIST
python experiments/mnist.py --name disvae

# CIFAR10
python experiments/cifar100_from_mnist.py --name disvae --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name disvae --data cifar100
```

### (Extra) Lucid

Running lucid visualisation on the trained VAE networks. In order to do this, first you need to run either experiment from (Claim 3) Disentanglement. Then, you can visualise run1 with lucid by pointing at the result folder, in which the script will also save the results.

Note that for this experiment, you need to first install the torch lucent module:
```
pip install torch-lucent
```

And then change a line in the module to make it work with MNIST and VAE. [This if branch](https://github.com/greentfrapp/lucent/blob/dev/lucent/optvis/render.py#L77) needs to be commented out. If
that's done, you can run the lucid experiment with


```
# Choose data from mnist, cifar10, cifar100
python experiments/lucid.py path/to/vae/folder --data mnist
```