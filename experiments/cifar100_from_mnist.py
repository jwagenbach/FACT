import argparse
import csv
import itertools
import logging
import os
from pathlib import Path
import time

import sys

sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms

from lfxai.explanations.examples import (InfluenceFunctions,
                                         NearestNeighbours,
                                         SimplEx,
                                         TracIn,
                                         CosineNearestNeighbours)
from lfxai.explanations.features import attribute_auxiliary, attribute_individual_dim
from lfxai.models.cifar100_models import (
    VAE,
    AutoEncoderCIFAR,
    ClassifierCIFAR,
    DecoderBurgess,
    DecoderCIFAR,
    EncoderBurgess,
    EncoderCIFAR,
)
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.models.pretext import Identity, Mask, RandomNoise
from lfxai.utils.datasets import MaskedCIFAR10, MaskedCIFAR100
from lfxai.utils.feature_attribution import generate_masks
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    similarity_rates,
    spearman_saliency,
)
from lfxai.utils.visualize import (
    correlation_latex_table,
    plot_pretext_saliencies,
    plot_pretext_top_example,
    plot_vae_saliencies,
    vae_box_plots,
)


def get_dataset(dataset: str, *args, **kwargs):
    if dataset.lower() == 'cifar10':
        return torchvision.datasets.CIFAR10(*args, **kwargs)
    elif dataset.lower() == 'cifar100':
        return torchvision.datasets.CIFAR100(*args, **kwargs)
    else:
        raise NameError('Unknown dataset')


def get_masked_dataset(dataset: str, *args, **kwargs):
    if dataset.lower() == 'cifar10':
        return MaskedCIFAR10(*args, **kwargs)
    elif dataset.lower() == 'cifar100':
        return MaskedCIFAR100(*args, **kwargs)
    else:
        raise NameError('Unknown dataset')


def consistency_feature_importance(
    dataset: str,
    random_seed: int = 1,
    batch_size: int = 256,
    dim_latent: int = 16,
    n_epochs: int = 100,
    inference: bool = False,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = 32  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]

    # Load MNIST
    data_dir = Path.cwd() / f"data/{dataset.lower()}"
    train_dataset = get_dataset(dataset, data_dir, train=True, download=True)
    test_dataset = get_dataset(dataset, data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderCIFAR(encoded_space_dim=dim_latent)
    decoder = DecoderCIFAR(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderCIFAR(encoder, decoder, dim_latent, pert).to(device)
    encoder.to(device)
    decoder.to(device)

    # Train the denoising autoencoder
    save_dir = Path.cwd() / f"results/cifar_vs_mnist/{dataset}/consistency_features"
    fig_folder = Path.cwd() / "figures"
    if not fig_folder.exists():
        os.makedirs(fig_folder)
    if not save_dir.exists():
        os.makedirs(save_dir)
    if not inference:
        autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)

    if inference:
        logging.info("Removing Integrated Gradients to save space in GPU memory.")
        attr_methods = {
            "Gradient Shap": GradientShap,
            "Saliency": Saliency,
            "Random": None,
        }
    else:
        attr_methods = {
            "Gradient Shap": GradientShap,
            "Integrated Gradients": IntegratedGradients,
            "Saliency": Saliency,
            "Random": None,
        }
    results_data = []
    baseline_features = torch.zeros((1, 3, W, W)).to(device)  # Baseline image for attributions
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(encoder,
                                       test_loader,
                                       device,
                                       attr_method(encoder),
                                       baseline_features)
        else:
            np.random.seed(random_seed)
            attr = np.random.randn(len(test_dataset), 3, W, W)

        for pert_percentage in pert_percentages:
            logging.info(f"Perturbing {pert_percentage}% of the features with {method_name}")
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[batch_id * batch_size:batch_id * batch_size + len(images)].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask * images
                pert_reps = encoder(images)
                rep_shift = torch.mean(torch.sum((original_reps - pert_reps)**2, dim=-1)).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    results_df = pd.DataFrame(results_data,
                              columns=["Method", "% Perturbed Pixels", "Representation Shift"])
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind", len(train_dataset.classes))
    sns.lineplot(data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method")
    plt.tight_layout()

    if inference:
        logging.info(f"Saving results in {fig_folder}")
        plt.savefig(fig_folder / f"claim1.1_{dataset}.pdf")
    else:
        logging.info(f"Saving results in {save_dir}")
        plt.savefig(save_dir / f"{dataset}_consistency_features.pdf")

    plt.close()


def consistency_examples(dataset: str,
                         random_seed: int = 1,
                         batch_size: int = 200,
                         dim_latent: int = 16,
                         n_epochs: int = 100,
                         subtrain_size: int = 1000,
                         inference: bool = False) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    data_dir = Path.cwd() / f"data/{dataset.lower()}"
    train_dataset = get_dataset(dataset, data_dir, train=True, download=True)
    test_dataset = get_dataset(dataset, data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderCIFAR(encoded_space_dim=dim_latent)
    decoder = DecoderCIFAR(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderCIFAR(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # Train the denoising autoencoder
    logging.info("Now fitting autoencoder")
    save_dir = Path.cwd() / f"results/cifar_vs_mnist/{dataset}/consistency_examples"
    fig_folder = Path.cwd() / "figures"
    if not fig_folder.exists():
        os.makedirs(fig_folder)
    if not save_dir.exists():
        os.makedirs(save_dir)
    if not inference:
        autoencoder.fit(device,
                        train_loader,
                        test_loader,
                        save_dir,
                        n_epochs,
                        checkpoint_interval=10)
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)
    autoencoder.train().to(device)

    n_classes = len(train_dataset.classes)
    idx_subtrain = [
        torch.nonzero(torch.Tensor(train_dataset.targets) == (n % n_classes))[n //
                                                                              n_classes].item()
        for n in range(subtrain_size)
    ]
    idx_subtest = [
        torch.nonzero(torch.Tensor(test_dataset.targets) == (n % n_classes))[n // n_classes].item()
        for n in range(subtrain_size)
    ]
    train_subset = Subset(train_dataset, idx_subtrain)
    test_subset = Subset(test_dataset, idx_subtest)
    subtrain_loader = DataLoader(train_subset, pin_memory=True, num_workers=2)
    subtest_loader = DataLoader(test_subset, pin_memory=True, num_workers=2)
    labels_subtrain = torch.cat([label for _, label in subtrain_loader])
    labels_subtest = torch.cat([label for _, label in subtest_loader])

    # Create a training set sampler with replacement for computing influence functions
    recursion_depth = 100
    train_sampler = RandomSampler(train_dataset,
                                  replacement=True,
                                  num_samples=recursion_depth * batch_size)
    train_loader_replacement = DataLoader(train_dataset,
                                          batch_size,
                                          sampler=train_sampler,
                                          pin_memory=True,
                                          num_workers=2)

    # Fitting explainers, computing the metric and saving everything
    mse_loss = torch.nn.MSELoss()

    if inference:
        logging.info(
            "In order to save 99% of computation, Influence Functions and TraceIn are ignored.")
        explainer_list = [
            SimplEx(autoencoder, mse_loss),
            NearestNeighbours(autoencoder, mse_loss),
            CosineNearestNeighbours(autoencoder, mse_loss)
        ]
    else:
        explainer_list = [
            InfluenceFunctions(autoencoder, mse_loss, save_dir / "if_grads"),
            TracIn(autoencoder, mse_loss, save_dir / "tracin_grads"),
            SimplEx(autoencoder, mse_loss),
            NearestNeighbours(autoencoder, mse_loss),
            CosineNearestNeighbours(autoencoder, mse_loss)
        ]
    frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    n_top_list = [int(frac * len(idx_subtrain)) for frac in frac_list]
    results_list = []
    for explainer in explainer_list:
        logging.info(f"Now fitting {explainer} explainer")
        attribution = explainer.attribute_loader(
            device,
            subtrain_loader,
            subtest_loader,
            train_loader_replacement=train_loader_replacement,
            recursion_depth=recursion_depth,
        )
        autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)
        sim_most, sim_least = similarity_rates(
            attribution, labels_subtrain, labels_subtest, n_top_list
        )
        results_list += [[str(explainer), "Most Important", 100 * frac, sim] for frac,
                         sim in zip(frac_list, sim_most)]
        results_list += [[str(explainer), "Least Important", 100 * frac, sim] for frac,
                         sim in zip(frac_list, sim_least)]
    results_df = pd.DataFrame(
        results_list,
        columns=[
            "Explainer",
            "Type of Examples",
            "% Examples Selected",
            "Similarity Rate",
        ],
    )
    if not inference:
        logging.info(f"Saving results in {save_dir}")
        results_df.to_csv(save_dir / "metrics.csv")
        sns.lineplot(
            data=results_df,
            x="% Examples Selected",
            y="Similarity Rate",
            hue="Explainer",
            style="Type of Examples",
            palette=sns.color_palette("colorblind", len(train_dataset.classes)),
        )
        plt.savefig(save_dir / "similarity_rates.pdf")
    else:
        logging.info(f"Saving results in {fig_folder}")
        sns.lineplot(
            data=results_df,
            x="% Examples Selected",
            y="Similarity Rate",
            hue="Explainer",
            style="Type of Examples",
            palette="colorblind",
        )
        plt.savefig(fig_folder / f"claim1.2_{dataset.lower()}.pdf")


def pretext_task_sensitivity(dataset: str,
                             random_seed: int = 1,
                             batch_size: int = 300,
                             n_runs: int = 5,
                             dim_latent: int = 16,
                             n_epochs: int = 100,
                             patience: int = 10,
                             subtrain_size: int = 1000,
                             n_plots: int = 10,
                             inference: bool = False) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mse_loss = torch.nn.MSELoss()

    # Load MNIST
    W = 32
    data_dir = Path.cwd() / f"data/{dataset.lower()}"
    train_dataset = get_dataset(dataset, data_dir, train=True, download=True)
    test_dataset = get_dataset(dataset, data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    X_train = []
    for x, _ in train_loader:
        X_train.append(x)
    X_train = torch.concat(X_train, dim=0)

    X_test = []
    for x, _ in test_loader:
        X_test.append(x)
    X_test = torch.concat(X_test, dim=0)

    n_classes = len(train_dataset.classes)
    idx_subtrain = [
        torch.nonzero(torch.Tensor(train_dataset.targets) == (n % n_classes))[n //
                                                                              n_classes].item()
        for n in range(subtrain_size)
    ]

    # Create saving directory
    save_dir = Path.cwd() / f"results/cifar_vs_mnist/{dataset}/pretext"
    fig_folder = Path.cwd() / "figures"
    if not fig_folder.exists():
        os.makedirs(fig_folder)
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    pretext_list = [Identity(), RandomNoise(noise_level=0.3), Mask(mask_proportion=0.2)]
    headers = [str(pretext) for pretext in pretext_list] + ["Classification"]  # Name of each task
    n_tasks = len(pretext_list) + 1
    feature_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    feature_spearman = np.zeros((n_runs, n_tasks, n_tasks))
    example_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    example_spearman = np.zeros((n_runs, n_tasks, n_tasks))

    for run in range(n_runs):
        feature_importance = []
        example_importance = []
        # Perform the experiment with several autoencoders trained on different pretext tasks.
        for pretext in pretext_list:
            # Create and fit an autoencoder for the pretext task
            name = f"{str(pretext)}-ae_run{run}"
            encoder = EncoderCIFAR(dim_latent)
            decoder = DecoderCIFAR(dim_latent)
            model = AutoEncoderCIFAR(encoder, decoder, dim_latent, pretext, name).to(device)
            if not inference:
                logging.info(f"Now fitting {name}")
                model.train()
                model.fit(device, train_loader, test_loader, save_dir, n_epochs, patience)
            model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
            model.eval().to(device)
            # Compute feature importance
            logging.info("Computing feature importance")
            baseline_image = torch.zeros((1, 3, 32, 32), device=device)
            gradshap = GradientShap(encoder)
            feature_importance.append(
                np.abs(
                    np.expand_dims(
                        attribute_auxiliary(encoder, test_loader, device, gradshap, baseline_image),
                        0,
                    )))
            # Compute example importance
            logging.info("Computing example importance")
            dknn = NearestNeighbours(model.cpu(), mse_loss, X_train)
            example_importance.append(
                np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0))

        # Create and fit a MNIST classifier
        name = f"Classifier_run{run}"
        encoder = EncoderCIFAR(dim_latent)
        classifier = ClassifierCIFAR(encoder,
                                     dim_latent,
                                     name,
                                     n_classes=len(train_dataset.classes)).to(device)
        if not inference:
            logging.info(f"Now fitting {name}")
            classifier.train()
            classifier.fit(device, train_loader, test_loader, save_dir, n_epochs, patience)
        classifier.eval().to(device)
        classifier.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
        baseline_image = torch.zeros((1, 3, 32, 32), device=device)
        # Compute feature importance for the classifier
        logging.info("Computing feature importance")
        gradshap = GradientShap(encoder)
        feature_importance.append(
            np.abs(
                np.expand_dims(
                    attribute_auxiliary(encoder, test_loader, device, gradshap, baseline_image),
                    0,
                )))
        # Compute example importance for the classifier
        logging.info("Computing example importance")
        dknn = NearestNeighbours(classifier.cpu(), mse_loss, X_train)
        example_importance.append(
            np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0))

        # Compute correlation between the saliency of different pretext tasks
        feature_importance = np.concatenate(feature_importance)
        feature_pearson[run] = np.corrcoef(feature_importance.reshape((n_tasks, -1)))
        feature_spearman[run] = spearmanr(feature_importance.reshape((n_tasks, -1)), axis=1)[0]
        example_importance = np.concatenate(example_importance)
        example_pearson[run] = np.corrcoef(example_importance.reshape((n_tasks, -1)))
        example_spearman[run] = spearmanr(example_importance.reshape((n_tasks, -1)), axis=1)[0]
        logging.info(
            f"Run {run} complete \n Feature Pearson \n {np.round(feature_pearson[run], decimals=2)}"
            f"\n Feature Spearman \n {np.round(feature_spearman[run], decimals=2)}"
            f"\n Example Pearson \n {np.round(example_pearson[run], decimals=2)}"
            f"\n Example Spearman \n {np.round(example_spearman[run], decimals=2)}")

        # Plot a couple of examples
        n_classes = len(train_dataset.classes)
        idx_plot = [
            torch.nonzero(torch.Tensor(test_dataset.targets) == (n % n_classes))[n //
                                                                                 n_classes].item()
            for n in range(n_plots)
        ]
        test_images_to_plot = [
            np.transpose(X_test[i].numpy().reshape(3, W, W), (1, 2, 0)) for i in idx_plot
        ]
        train_images_to_plot = [
            np.transpose(X_train[i].numpy().reshape(3, W, W), (1, 2, 0)) for i in idx_subtrain
        ]
        fig_features = plot_pretext_saliencies(test_images_to_plot,
                                               feature_importance[:, idx_plot, :, :, :],
                                               headers,
                                               n_classes=len(train_dataset.classes))
        fig_features.savefig(save_dir / f"saliency_maps_run{run}.pdf")
        plt.close(fig_features)
        fig_examples = plot_pretext_top_example(train_images_to_plot,
                                                test_images_to_plot,
                                                example_importance[:, idx_plot, :],
                                                headers)
        plt.tight_layout()
        if inference:
            if run == n_runs - 1:
                fig_examples.savefig(save_dir / f"cifar10_top_examples.pdf")
        else:
            fig_examples.savefig(save_dir / f"top_examples_run{run}.pdf")
        plt.close(fig_features)

    # Compute the avg and std for each metric
    feature_pearson_avg = np.round(np.mean(feature_pearson, axis=0), decimals=2)
    feature_pearson_std = np.round(np.std(feature_pearson, axis=0), decimals=2)
    feature_spearman_avg = np.round(np.mean(feature_spearman, axis=0), decimals=2)
    feature_spearman_std = np.round(np.std(feature_spearman, axis=0), decimals=2)
    example_pearson_avg = np.round(np.mean(example_pearson, axis=0), decimals=2)
    example_pearson_std = np.round(np.std(example_pearson, axis=0), decimals=2)
    example_spearman_avg = np.round(np.mean(example_spearman, axis=0), decimals=2)
    example_spearman_std = np.round(np.std(example_spearman, axis=0), decimals=2)

    # Format the metrics in Latex tables
    if inference:
        with open(fig_folder / f"claim2.1_{dataset}.tex", "w") as f:
            f.write(correlation_latex_table(feature_pearson_avg, feature_pearson_std, headers))
            f.write("\n")
        with open(fig_folder / f"claim2.2_{dataset}.tex", "w") as f:
            f.write(correlation_latex_table(example_pearson_avg, example_pearson_std, headers))
            f.write("\n")
    else:
        with open(save_dir / "tables.tex", "w") as f:
            for corr_avg, corr_std in zip(
                [
                    feature_pearson_avg,
                    feature_spearman_avg,
                    example_pearson_avg,
                    example_spearman_avg,
                ],
                [
                    feature_pearson_std,
                    feature_spearman_std,
                    example_pearson_std,
                    example_spearman_std,
                ],
            ):
                f.write(correlation_latex_table(corr_avg, corr_std, headers))
                f.write("\n")


def disvae_feature_importance(dataset: str,
                              random_seed: int = 1,
                              batch_size: int = 300,
                              n_plots: int = 20,
                              n_runs: int = 5,
                              dim_latent: int = 16,
                              n_epochs: int = 100,
                              beta_list: list = [1, 5, 10],
                              inference: bool = False) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    W = 32
    img_size = (3, W, W)
    data_dir = Path.cwd() / f"data/{dataset.lower()}"
    train_dataset = get_dataset(dataset, data_dir, train=True, download=True)
    test_dataset = get_dataset(dataset, data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.Resize(W), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(W), transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    # Create saving directory
    save_dir = Path.cwd() / f"results/cifar_vs_mnist/{dataset}/vae"
    fig_folder = Path.cwd() / "figures"
    if not fig_folder.exists():
        os.makedirs(fig_folder)
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [
        BetaHLoss(rec_dist='gaussian'),
        BtcvaeLoss(rec_dist='gaussian', is_mss=False, n_data=len(train_dataset))
    ]
    metric_list = [pearson_saliency]
    metric_names = ["Pearson Correlation"]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, "w") as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=",", fieldnames=headers)
            dw.writeheader()

    for beta, loss, run in itertools.product(beta_list, loss_list, range(1, n_runs + 1)):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss.beta = beta
        name = f"{str(loss)}-vae_beta{beta}_run{run}"
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name).to(device)
        if not inference:
            logging.info(f"Now fitting {name}")
            model.train()
            model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=True)
        model.eval()

        # Compute test-set saliency and associated metrics
        baseline_image = torch.zeros((1, 3, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(encoder.mu,
                                                dim_latent,
                                                test_loader,
                                                device,
                                                gradshap,
                                                baseline_image)
        metrics = compute_metrics(attributions, metric_list)
        results_str = "\t".join(
            [f"{metric_names[k]} {metrics[k]:.2g}" for k in range(len(metric_list))])
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([str(loss), beta] + metrics)

        # Plot a couple of examples
        if not inference:
            n_classes = len(train_dataset.classes)
            plot_idx = [
                torch.nonzero(
                    torch.Tensor(test_dataset.targets) == (n % n_classes))[n // n_classes].item()
                for n in range(n_plots)
            ]
            images_to_plot = [
                np.transpose(test_dataset[i][0].numpy().reshape(3, W, W), (1, 2, 0))
                for i in plot_idx
            ]
            fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx], n_dim=dim_latent)
            fig.savefig(save_dir / f"{name}.pdf")
            plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    if inference:
        logging.info(f"Saving results in {fig_folder}")
        fig.savefig(fig_folder / f"claim3_{dataset}.pdf")
    else:
        logging.info(f"Saving results in {save_dir}")
        fig.savefig(save_dir / "metric_box_plots.pdf")
    plt.close(fig)


def roar_test(
    dataset: str,
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 16,
    n_epochs: int = 100,
) -> None:
    # Initialize seed and device
    logging.info("Welcome in the ROAR test experiments")
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    remove_percentages = [10, 20, 50, 70, 100]

    # Load MNIST
    W = 32  # Image width = height
    data_dir = Path.cwd() / f"data/{dataset.lower()}"
    train_dataset = get_dataset(dataset, data_dir, train=True, download=True)
    test_dataset = get_dataset(dataset, data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)
    save_dir = Path.cwd() / f"results/cifar_vs_mnist/{dataset}/roar_test"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Initialize encoder, decoder and autoencoder wrapper
    pert = Identity()
    encoder = EncoderCIFAR(encoded_space_dim=dim_latent)
    decoder = DecoderCIFAR(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderCIFAR(encoder, decoder, dim_latent, pert, name="model_initial")
    autoencoder.save(save_dir)
    encoder.to(device)
    decoder.to(device)

    # Train the denoising autoencoder
    logging.info("Training the initial autoencoder")
    autoencoder = AutoEncoderCIFAR(encoder, decoder, dim_latent, pert, name="model")
    autoencoder.load_state_dict(torch.load(save_dir / "model_initial.pt"), strict=False)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)
    original_test_performance = autoencoder.test_epoch(device, test_loader)

    # Create dictionaries to store feature importance and shift induced by perturbations
    explainer_dic = {
        "Gradient Shap": GradientShap(encoder),
        "Integrated Gradients": IntegratedGradients(encoder),
        "Random": None,
    }
    baseline_features = torch.zeros((1, 3, W, W)).to(device)  # Baseline image for attributions
    results_data = []

    for explainer_name in explainer_dic:
        logging.info(f"Computing feature importance with {explainer_name}")
        results_data.append([explainer_name, 0, original_test_performance])
        if explainer_dic[explainer_name] is not None:
            attr = attribute_auxiliary(
                encoder,
                train_loader,
                device,
                explainer_dic[explainer_name],
                baseline_features,
            )
        else:  # Random attribution
            attr = np.random.randn(len(train_dataset), 3, W, W)
        for remove_percentage in remove_percentages:
            mask_size = int(remove_percentage * (W**2) / 100)
            torch.random.manual_seed(random_seed)
            logging.info(
                f"Retraining an autoencoder with {remove_percentage}% pixels masked by {explainer_name}"
            )
            masks = generate_masks(attr, mask_size)
            masked_train_set = get_masked_dataset(dataset, data_dir, True, masks)
            masked_train_set.transform = train_transform
            masked_train_loader = DataLoader(masked_train_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=2)
            encoder = EncoderCIFAR(encoded_space_dim=dim_latent)
            decoder = DecoderCIFAR(encoded_space_dim=dim_latent)
            autoencoder_name = f"model_{explainer_name}_mask{mask_size}"
            autoencoder = AutoEncoderCIFAR(encoder,
                                           decoder,
                                           dim_latent,
                                           pert,
                                           name=autoencoder_name)
            autoencoder.load_state_dict(torch.load(save_dir / "model_initial.pt"), strict=False)
            encoder.to(device)
            decoder.to(device)
            autoencoder.fit(device, masked_train_loader, test_loader, save_dir, n_epochs)
            autoencoder.load_state_dict(torch.load(save_dir / (autoencoder_name + ".pt")),
                                        strict=False)
            test_performance = autoencoder.test_epoch(device, test_loader)
            results_data.append([explainer_name, remove_percentage, test_performance])

    logging.info(f"Saving the plot in {str(save_dir)}")
    results_df = pd.DataFrame(results_data,
                              columns=["Method", "% of features removed", "Test Loss"])
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind", len(train_dataset.classes))
    sns.lineplot(data=results_df, x="% of features removed", y="Test Loss", hue="Method")
    plt.tight_layout()
    plt.savefig(save_dir / "roar.pdf")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="disvae")
    parser.add_argument("--data", type=str, default="cifar100")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--inference", action='store_true')
    args = parser.parse_args()

    n_epochs = int(not args.debug) * 98 + 2
    start = time.time()

    if args.name == "disvae":
        disvae_feature_importance(args.data,
                                  n_runs=args.n_runs,
                                  batch_size=args.batch_size,
                                  random_seed=args.random_seed,
                                  n_epochs=n_epochs,
                                  inference=args.inference)
    elif args.name == "pretext":
        pretext_task_sensitivity(args.data,
                                 n_runs=args.n_runs,
                                 batch_size=args.batch_size,
                                 random_seed=args.random_seed,
                                 n_epochs=n_epochs,
                                 inference=args.inference)
    elif args.name == "consistency_features":
        consistency_feature_importance(args.data,
                                       batch_size=args.batch_size,
                                       random_seed=args.random_seed,
                                       n_epochs=n_epochs,
                                       inference=args.inference)
    elif args.name == "consistency_examples":
        consistency_examples(args.data,
                             batch_size=args.batch_size,
                             random_seed=args.random_seed,
                             n_epochs=n_epochs,
                             inference=args.inference)
    elif args.name == "roar_test":
        n_epochs = int(not args.debug) * 8 + 2
        roar_test(args.data,
                  batch_size=args.batch_size,
                  random_seed=args.random_seed,
                  n_epochs=n_epochs)
    else:
        raise ValueError("Invalid experiment name")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
