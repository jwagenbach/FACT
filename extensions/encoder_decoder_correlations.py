import numpy as np
import torch
from captum.attr import Attribution
from lfxai.explanations.features import attribute_auxiliary
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class EncoderDecoderComparison:
    """This class handles experiments for comparing feature importance metrics
    of the encoder (which maps from input to latent space) and a full model (which maps from input to prediction)"""
    def __init__(self,
                 model_name: str,
                 full_model: torch.nn.Module,
                 encoder: torch.nn.Module,
                 full_model_attributer: Attribution,
                 encoder_attributer: Attribution,
                 dataset: str,
                 data_directory='./data',
                 model_directory='../TrainedModels',
                 device='cpu'
                 ):
        allowed_datasets = ['MNIST', 'CIFAR']
        assert dataset in allowed_datasets, f"Dataset must be one of {allowed_datasets}"

        self.model_name = model_name
        self.full_model = full_model
        self.encoder = encoder
        self.dataset_name = dataset
        self.data_directory = data_directory
        self.model_directory = model_directory

        # Instantiate attributers
        self.full_model_attributer = full_model_attributer
        self.encoder_attibuter = encoder_attributer

        self.batch_size = 128
        self.device = device

        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = self.get_data_and_loaders()
        self.baseline_image = self.get_baseline_image()
        print("Calculating Attributions")

        self.encoder_attributions, self.full_attributions = self.get_all_attributions()
        print("attributions calculated")

    def get_data_and_loaders(self):
        if self.dataset_name == 'MNIST':
            return self._get_MNIST_data()
        elif self.dataset_name == 'CIFAR':
            return self._get_CIFAR_data()
        else:
            raise ValueError()

    def get_baseline_image(self):

        dimensions = {'CIFAR': 32,
                      'MNIST': 28}

        dimension = dimensions[self.dataset_name]
        baseline_image = torch.zeros((1, 1, dimension, dimension), device=self.device)

        return baseline_image

    def get_all_attributions(self):
        encoder_attributions = attribute_auxiliary(
            self.encoder, self.test_loader, self.device, self.encoder_attibuter, self.baseline_image
        )

        # Note that this is the correct thing to do here because the classifier outputs probabilities so we are taking a soft sum
        full_model_attributions = attribute_auxiliary(
            self.full_model, self.test_loader, self.device, self.full_model_attributer, self.baseline_image
        )

        # Cast each one to absolute value, since we're not interested in the direction on the hidden space
        encoder_attributions = np.abs(encoder_attributions)
        full_model_attributions = np.abs(full_model_attributions)

        # Normalise each one to have variance 1 - doesnt affect downstream analysis but makes scales much more interpretable
        encoder_attributions = (encoder_attributions)/ np.std(encoder_attributions)
        full_model_attributions = (full_model_attributions) / np.std(full_model_attributions)

        return encoder_attributions, full_model_attributions

    def _get_MNIST_data(self):
        data_dir = "data/mnist"
        shared_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = MNIST(data_dir,
                              train=True,
                              download=True,
                              transform=shared_transform
                              )

        test_dataset = MNIST(data_dir,
                             train=False,
                             download=True,
                             transform=shared_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False
                                                  )

        return train_dataset, test_dataset, train_loader, test_loader

    def _get_CIFAR_data(self):
        raise NotImplemented()

    def plot_mean_pixel_attributions(self):
        fig, ax = plt.subplots(1, 2, sharey='row', figsize=[15, 6])
        ax[0].imshow(self.encoder_attributions.mean(axis=0).squeeze(), cmap='gray_r')
        ax[0].set_title('Encoder Saliency Map')

        ax[1].imshow(self.full_attributions.mean(axis=0).squeeze(), cmap='gray_r')
        ax[1].set_title('Classifier Saliency Map')


    def _image_pearson(self, i: int, mask_dead_pixels=True):
        """Returns the pearson correlation coefficient for the saliency maps of
        the encoder and the full classifier of one image.

        If the :mask_dead_pixels: flag is True, then we will only consider
        """
        X, y = self.test_dataset[i]
        X = X.squeeze()

        a_enc_i = self.encoder_attributions[i].squeeze()
        a_full_i = self.full_attributions[i].squeeze()

        if mask_dead_pixels:
            mask = (X == 0)
            a_enc_i = a_enc_i[~mask]
            a_full_i = a_full_i[~mask]

        # corrcoef returns a covariance matrix, so we have to take the off diagonal element
        rho = np.corrcoef(a_enc_i.flatten(), a_full_i.flatten())[0, 1]

        return rho

    def get_all_image_pearsons(self, mask_dead_pixels=True):
        """Returns an array of the pearson correlation coefficients of t"""
        rhos = []
        for i in range(len(self.test_dataset)):
            rho = self._image_pearson(i, mask_dead_pixels=mask_dead_pixels)
            rhos.append(rho)

        rhos = np.array(rhos)

        return rhos

