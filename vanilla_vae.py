"""
Thanks to https://github.com/AntixK/PyTorch-VAE
"""

import torch
#from models import BaseVAE
from typing import List, TypeVar
from torch import nn
#from .types_ import *
Tensor = TypeVar('torch.tensor')
class VanillaVAE(nn.Module):

    def __init__(self,
                 in_size = [96,96,96],
                 in_channels = 3,
                 out_channels = 3,
                 latent_dim = 128,
                 hidden_dims = None):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
            self.hidden = [32, 64, 128, 256]
        

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.oneDim = int(in_size[0] / 2 ** len(hidden_dims))
        self.finalSize = int(self.oneDim ** 3)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.finalSize, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.finalSize, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.finalSize)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv3d(hidden_dims[i],
                              out_channels= hidden_dims[i + 1],
                              kernel_size=3,padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv3d(hidden_dims[-1],
                              out_channels= hidden_dims[-1],
                              kernel_size=3,padding=1),
                            nn.BatchNorm3d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1], out_channels= out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden[-1], self.oneDim, self.oneDim, self.oneDim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]