import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr._utils.visualization import visualize_image_attr
from tabulate import tabulate


def plot_image_saliency(image: torch.Tensor, saliency: torch.Tensor):
    image_np = image.permute((1, 2, 0)).cpu().numpy()
    saliency_np = saliency.permute((1, 2, 0)).cpu().numpy()
    visualize_image_attr(saliency_np, image_np)


def plot_pretext_saliencies(images: list,
                            saliency: np.ndarray,
                            pretext_names: list,
                            n_classes: int = 10) -> plt.Figure:
    W = saliency.shape[-1]
    n_pretext = len(pretext_names)
    n_plots = len(images)
    cblind_palette = sns.color_palette("colorblind")
    fig, axs = plt.subplots(ncols=n_pretext, nrows=n_plots, figsize=(3 * n_pretext, 3 * n_plots))
    for example_id in range(n_plots):
        for pretext_id, pretext in enumerate(pretext_names):
            sub_saliency = saliency[pretext_id, example_id]
            ax = axs[example_id, pretext_id]
            if len(images[example_id].shape) == 2:
                ax.imshow(images[example_id], cmap="gray", zorder=1)
            else:
                ax.imshow(images[example_id])
            ax.axis("off")
            sns.heatmap(
                np.reshape(sub_saliency, (-1, W, W)).mean(axis=0),
                linewidth=0,
                xticklabels=False,
                yticklabels=False,
                ax=ax,
                cmap=sns.light_palette(cblind_palette[pretext_id], n_classes, as_cmap=True),
                cbar=False,
                alpha=0.8,
                zorder=2,
                vmin=0,
            )
            ax.set_title(f"Saliency {pretext}")
    return fig


def plot_pretext_top_example(
    train_images: list,
    test_images: list,
    example_importance: np.ndarray,
    pretext_names: list,
) -> plt.Figure:
    n_pretext = len(pretext_names)
    n_plots = len(test_images)
    fig, axs = plt.subplots(
        ncols=n_pretext + 1, nrows=n_plots, figsize=(3 * (n_pretext + 1), 3 * n_plots)
    )
    for example_id in range(n_plots):
        ax = axs[example_id, 0]
        if len(test_images[example_id].shape) == 2:
            ax.imshow(test_images[example_id], cmap="gray")
        else:
            ax.imshow(test_images[example_id])
        ax.axis("off")
        ax.set_title("Test Image", fontdict={'fontsize': 20, 'fontweight': 'medium'})
        for pretext_id, pretext in enumerate(pretext_names):
            top_id = np.argmax(example_importance[pretext_id, example_id, :])
            ax = axs[example_id, pretext_id + 1]
            if len(train_images[top_id].shape) == 2:
                ax.imshow(train_images[top_id], cmap="gray")
            else:
                ax.imshow(train_images[top_id])
            ax.axis("off")
            ax.set_title(f"{pretext}", fontdict={'fontsize': 20, 'fontweight': 'medium'})
    return fig


def plot_vae_saliencies(images: list, saliency: np.ndarray, n_dim: int = 4) -> plt.Figure:
    W = saliency.shape[-1]
    n_plots = len(saliency)
    dim_latent = saliency.shape[1]
    cblind_palette = sns.color_palette("colorblind", n_dim)
    fig, axs = plt.subplots(
        ncols=dim_latent + 1, nrows=n_plots, figsize=(3 * (dim_latent + 1), 3 * n_plots)
    )
    for example_id in range(n_plots):
        max_saliency = np.max(saliency[example_id])
        ax = axs[example_id, 0]
        if len(images[example_id].shape) == 2:
            ax.imshow(images[example_id], cmap="gray")
        else:
            ax.imshow(images[example_id])
        ax.axis("off")
        ax.set_title("Original Image")
        for dim in range(dim_latent):
            sub_saliency = saliency[example_id, dim]
            ax = axs[example_id, dim + 1]
            sns.heatmap(
                np.reshape(sub_saliency, (-1, W, W)).mean(axis=0),
                linewidth=0,
                xticklabels=False,
                yticklabels=False,
                ax=ax,
                cmap=sns.light_palette(cblind_palette[dim], n_dim, as_cmap=True),
                cbar=False,
                alpha=1,
                zorder=2,
                vmin=0,
                vmax=max_saliency,
            )
            ax.set_title(f"Saliency Dimension {dim+1}")
    return fig


def vae_box_plots(df: pd.DataFrame, metric_names: list, n_classes: int = 10) -> plt.Figure:
    fig, axs = plt.subplots(ncols=1, nrows=len(metric_names), figsize=(6, 4 * len(metric_names)))
    for id_metric, metric in enumerate(metric_names):
        sns.boxplot(
            data=df,
            x="Beta",
            y=metric,
            hue="Loss Type",
            palette=sns.color_palette("colorblind", n_classes),
            ax=axs[id_metric],
        )
    return fig


def correlation_latex_table(corr_avg: np.ndarray, corr_std: np.ndarray, headers: list) -> str:
    table = [[headers[i]] +
             [f"${corr_avg[i,j]} \\pm {corr_std[i,j]}$"
              for j in range(corr_avg.shape[0])]
             for i in range(corr_avg.shape[0])]
    return tabulate(table, tablefmt="latex_raw", headers=headers)
