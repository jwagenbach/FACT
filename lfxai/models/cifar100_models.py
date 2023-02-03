from .images import *


class EncoderCIFAR(nn.Module):

    def __init__(self, encoded_space_dim=128):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # 32
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 16
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 8
            nn.LayerNorm([128, 8, 8]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 4
            nn.ReLU(True),
        )
        self.encoder_lin = nn.Sequential(nn.Linear(128 * 4 * 4, 128),
                                         nn.ReLU(True),
                                         nn.Linear(128, encoded_space_dim))
        self.flatten = nn.Flatten(start_dim=1)
        self.encoded_space_dim = encoded_space_dim

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DecoderCIFAR(nn.Module):

    def __init__(self, encoded_space_dim=128):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 4 * 4 * 128),
            nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16
            nn.LayerNorm([32, 16, 16]),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 32
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoderCIFAR(AutoEncoderMnist):

    def __init__(self, encoder: EncoderCIFAR, decoder: DecoderCIFAR, *args, **kwargs):
        super().__init__(encoder, decoder, *args, **kwargs)


class ClassifierCIFAR(ClassifierMnist):

    def __init__(self, encoder: EncoderCIFAR, *args, n_classes=100, **kwargs):
        super().__init__(encoder, *args, n_classes=n_classes, **kwargs)


class VarEncoderCIFAR(nn.Module):

    def __init__(self, _: int = 64, latent_dims: int = 10):
        super(VarEncoderMnist, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = EncoderCIFAR(latent_dims * 2)

    def forward(self, x):
        x = self.encoder(x)
        x_mu = x[:self.latent_dims]
        x_logvar = x[self.latent_dims:]
        return x_mu, x_logvar


class VarDecoderCIFAR(DecoderCIFAR):

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)


class BetaVaeCIFAR(BetaVaeMnist):

    def __init__(self, latent_dims: int = 10, beta: int = 1):
        super().__init__(latent_dims, beta)
        self.encoder = VarEncoderCIFAR(latent_dims=latent_dims)
        self.decoder = VarDecoderCIFAR(latent_dims=latent_dims)


class BetaTcVaeCIFAR(BetaTcVaeMnist):

    def __init__(self, latent_dims: int = 10, beta: int = 1):
        super().__init__(latent_dims, beta)
        self.encoder = VarEncoderCIFAR(latent_dims=latent_dims)
        self.decoder = VarDecoderCIFAR(latent_dims=latent_dims)


class ClassifierLatentCIFAR(EncoderCIFAR):

    def __init__(self, latent_dims: int):
        super().__init__(latent_dims)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
