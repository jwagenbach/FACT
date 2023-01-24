import os

import numpy as np
import torch
from lfxai.explanations.features import attribute_auxiliary
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from lfxai.models.images import ClassifierMnist, EncoderMnist

import copy


class EncoderDecoderComparison:
    """This class handles experiments for comparing feature importance metrics
    of the encoder (which maps from input to latent space) and a full model (which maps from input to prediction)"""

    def __init__(self,
                 model_name: str,
                 attributer_factory,
                 data_directory='./data',
                 model_directory='../TrainedModels',
                 device='cpu'
                 ):
        """

        :param model_name: An arbitrary name that will be used to
        :param attributer_factory: This should be the __init__ method of a Captum Attribute subclass,
        :param data_directory: directory where data will be stored/loaded
        :param model_directory: directory where the trained models are stored.
        :param device: 'cpu' or 'gpu', depending on if cuda is available.
        """

        # Metadata
        self.model_name = model_name
        self.dataset_name = 'MNIST' # TODO need to expand this if we implement CIFAR
        self.data_directory = data_directory
        self.device = device

        self.attributer_factory = attributer_factory

        # Constants
        self.batch_size = 128
        self.dim_latent = 4

        # Load data
        print("Loading data...")
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = self.get_data_and_loaders()
        self.baseline_image = self.get_baseline_image()

        print("Loading Models and Calculating Attributions...")
        self.models = self._load_models(os.path.join(model_directory, self.dataset_name))

        print("Complete")

    def _make_attributer(self, model):
        raise NotImplementedError("Implement on subclass")

    def view_available_models(self):
        print(self.models.keys())

    def get_data_and_loaders(self):
        if self.dataset_name == 'MNIST':
            return self._get_MNIST_data()
        elif self.dataset_name == 'CIFAR':
            return self._get_CIFAR_data()
        else:
            raise ValueError()

    def _load_models(self, model_directory):
        models = [m for m in os.listdir(model_directory) if m.endswith('.pt')]

        out = {}
        for m in models:
            name = m.rstrip('.pt').split('_')[1]
            path = os.path.join(model_directory, m)
            print(f"Loading model in {path}")
            encoder = EncoderMnist(self.dim_latent)

            classifier = ClassifierMnist(encoder, self.dim_latent, name)
            classifier.load_state_dict(torch.load(path), strict=True)

            encoder = copy.deepcopy(classifier.encoder)  # Copies the params to the encoder variable

            # make one attributer for each model
            classifier_attributer = self.attributer_factory(classifier)
            encoder_attributer = self.attributer_factory(encoder)

            out[name] = {
                'full_model': {
                    'model': classifier,
                    'attributer': classifier_attributer,
                    'attributions': self._calculate_attributions(classifier, classifier_attributer)
                },
                'encoder': {
                    'model': encoder,
                    'attributer': encoder_attributer,
                    'attributions': self._calculate_attributions(encoder, encoder_attributer)}
            }

        return out

    def get_baseline_image(self):

        dimensions = {'CIFAR': 32,
                      'MNIST': 28}

        dimension = dimensions[self.dataset_name]
        baseline_image = torch.zeros((1, 1, dimension, dimension), device=self.device)

        return baseline_image

    def _calculate_attributions(self, model, attributer):
        attributions = attribute_auxiliary(
            model, self.test_loader, self.device, attributer, self.baseline_image
        )

        # Cast each one to absolute value, since we're not interested in the direction on the hidden space
        attributions = np.abs(attributions)

        # Normalise each one to have variance 1 - doesnt affect downstream analysis but makes scales much more interpretable
        attributions = (attributions) / np.std(attributions)

        return attributions

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


    def _image_pearson(self, i: int, encoder_attributions, full_attributions, mask_dead_pixels=True):
        """Returns the pearson correlation coefficient for the saliency maps of
        the encoder and the full classifier of one image.

        If the :mask_dead_pixels: flag is True, then we will only consider
        """

        a_enc_i = encoder_attributions[i].squeeze()
        a_full_i = full_attributions[i].squeeze()

        if mask_dead_pixels:
            X, y = self.test_dataset[i]
            X = X.squeeze()
            mask = (X == 0)
            a_enc_i = a_enc_i[~mask]
            a_full_i = a_full_i[~mask]

        # corrcoef returns a covariance matrix, so we have to take the off diagonal element
        rho = np.corrcoef(a_enc_i.flatten(), a_full_i.flatten())[0, 1]

        return rho

    def get_image_pearsons_for_one_model(self, encoder_attributions, full_attributions, mask_dead_pixels=True):
        """Returns an array of the pearson correlation coefficients of t"""
        rhos = []
        for i in range(len(self.test_dataset)):
            rho = self._image_pearson(i, encoder_attributions, full_attributions, mask_dead_pixels=mask_dead_pixels)
            rhos.append(rho)

        rhos = np.array(rhos)

        return rhos

    def get_all_model_pearsons(self, mask_dead_pixels):
        res = [] # List of np arrays, to be concatenated down rwos
        for model_name, model_data in self.models.items():
            full_model_attributions = model_data['full_model']['attributions']
            encoder_attributions = model_data['encoder']['attributions']

            rhos = self.get_image_pearsons_for_one_model(encoder_attributions,
                                                         full_model_attributions,
                                                         mask_dead_pixels)

            res.append(rhos)

        out = np.row_stack(res)
        return out

