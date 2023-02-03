import sys
import argparse
import os
import torch
from lucent.optvis import render, param, objectives
from matplotlib import pyplot as plt

sys.path.append('./')

from lfxai.models.images import VAE, EncoderBurgess, DecoderBurgess
from lfxai.models.losses import BetaHLoss

BETAS = [1, 5, 10]
LATENT_DIM = 3


def visualize(model_path: str, beta: int, data: str, iters: int):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dim_latent = 3 if data == 'mnist' else 16
    img_size = (1, 32, 32) if data == 'mnist' else (3, 32, 32)
    W = img_size[-1]
    encoder = EncoderBurgess(img_size, dim_latent, leaky=True).to(device)

    # Load
    model_state_dict = torch.load(model_path)
    encoder_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('encoder.'):
            encoder_state_dict[key[len('encoder.'):]] = value
    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder = encoder.eval()

    images = []
    for dim in range(0, dim_latent * 2, 2):
        param_f = lambda: param.image(W, channels=img_size[0])
        obj = objectives.channel("mu_logvar_gen", dim)
        img = render.render_vis(encoder,
                                obj,
                                param_f,
                                preprocess=False,
                                thresholds=(iters,),
                                progress=False,
                                verbose=False,
                                fixed_image_size=True,
                                transforms=[],
                                show_image=False)[0][0]

        # NOTE
        # In order to make this work with the lucent module, you need to comment
        # out this if statement from the module
        # https://github.com/greentfrapp/lucent/blob/dev/lucent/optvis/render.py#L77
        images.append(img)

    return images


def plot(save_folder: str, results: list, data: str):

    dims = 3 if data == 'mnist' else 16
    fig, axs = plt.subplots(nrows=len(BETAS), ncols=dims)

    for i, beta in enumerate(BETAS):
        for dim in range(dims):
            ax = axs[i, dim]
            result = results[i][dim]
            if data == 'mnist':
                ax.imshow(result, cmap='gray')
            else:
                ax.imshow(result, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, beta in zip(axs[:, 0], BETAS):
        ax.set_ylabel(r'$\beta$=' + str(beta), rotation=90, size='large')

    for ax, dim in zip(axs[0], list(range(dims))):
        ax.set_title(f"{dim}. unit")

    plt.savefig(os.path.join(save_folder, 'lucid.pdf'), format="pdf", bbox_inches="tight")
    plt.show()


def main(folder: str, iters: int, data: str):

    filepaths = {beta: os.path.join(folder, f'Beta-vae_beta{beta}_run1.pt') for beta in BETAS}

    # Run
    results = []
    for beta, filepath in filepaths.items():
        images = visualize(filepath, beta, data, iters)
        results.append(images)

    # Plot
    plot(folder, results, data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help='path to models')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--data", type=str, default='mnist')
    args = parser.parse_args()

    n_iters = 2 if args.debug else 1000

    main(args.path, n_iters, args.data)
