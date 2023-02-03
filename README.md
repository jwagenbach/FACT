# Reproducing "Label-Free Explainability for Unsupervised Models"

This repository is a reproduction of the [Label-Free Explainability for Unsupervised Models](https://arxiv.org/abs/2203.01928). It is heavily based on the authors originally code, which is available [here](https://github.com/JonathanCrabbe/Label-Free-XAI).

## Setup

We've saved the conda environment we used, so it can be reinstalled with
```
conda env create --file environment.yml
```

## Experiments

The structure of this repository can best be understood in relation to our accompanying report.

In the below, we have indicated how each experiment in the Methodology (section 3.1 of the report) can be run using this codebase.

Because the additional experiments for CIFAR10 and CIFAR100 were similar in structure to the original experiments, we have included the code for these experiments alongside the original ones. 

### (Claim 1.1) Feature importance consistency

These experiments produce Figure 1 in the report, which shows how masking important features affects representation shift in a number of datasets.

The code for each experiment is contained in the following scripts (one for each dataset). Run each of the following terminal commands to obtain the relevant results.

Each script will save the relevant figure into a directory under ./results/<dataset_name>/consistency_features

```
# MNIST
python experiments/mnist.py --name consistency_features

# ECG5000
python experiments/ecg5000.py --name consistency_features

# CIFAR10
python experiments/cifar100_from_mnist.py --name consistency_features --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name consistency_features --data cifar100
```

#### (Claim 1.2) Example importance consistency 

These experiments produce Figure 2 in the report, which show how similarity rates are affected by example importance metrics.

The code for each experiment is contained in the following scripts (one for each dataset). Run each of the following terminal commands to obtain the relevant results.

Each script will save the relevant figure into a directory under ./results/<dataset_name>/consistency_examples

```
# MNIST
python experiments/mnist.py --name consistency_examples

# ECG5000
python experiments/ecg5000.py --name consistency_examples

# CIFAR10
python experiments/cifar100_from_mnist.py --name consistency_examples --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name consistency_examples --data cifar100
```

### (Claims 2.1 and 2,2): Correlation of feature and example importance scores across pretext tasks

These scripts produce Tables 2, 3 and 4 from the report, which show the Pearson correlation of feature importance and example importance scores when latent representations are trained under different pretexts. 

These script outputs a latex table file under ./results/<script_name>/pretext, in which the first table refers to the feature importance Pearson correlation, and the third one is the same for example importance. The other two tables measure the Spearman (rank) correlation, which was not used in our report, but were produced by the authors' original code.

```
# MNIST
python experiments/mnist.py --name pretext

# CIFAR10
python experiments/cifar100_from_mnist.py --name pretext --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name pretext --data cifar100
```

### (Claim 3) Disentanglement

These scripts produce Figure 3, which shows how the Pearson correlation of feature importance scares between the latent units of disentangled VAEs.

We have additionally implemented this analysis on CIFAR10 and CIFAR100, although we did not have room to include this in our final report.

```
# MNIST
python experiments/mnist.py --name disvae

# Dsprites
python experiments/dsprites.py

# CIFAR10
python experiments/cifar100_from_mnist.py --name disvae --data cifar10

# CIFAR100
python experiments/cifar100_from_mnist.py --name disvae --data cifar100
```

#### Claim 3 extension: Lucid

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

## Additional Experiment - Comparison of Unsupervised and Supervised Feature Importance

This code is used to produce Figure 5, which shows how feature importance correlates between latent encoders and full models. 

The analysis is best run using the notebook found in extensions/encoder_decoder_correlations/first_class_encoder_vs_decoder_correlations.ipynb.

This notebook imports its codebase from extensions/encoder_decoder_correlations/encoder_decoder_correlations.py.
