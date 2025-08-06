"""
This module provides a PyTorch version of the PixelVAE model used in:
Storrs, K. R., Anderson, B. L., & Fleming, R. W. (2021).
Unsupervised learning predicts human perception and misperception of gloss.
Nature Human Behaviour, 5(10), 1402–1417. https://doi.org/10.1038/s41562-021-01097-6


The original implementation (TF) can be found here:
https://github.com/ermongroup/Generalized-PixelVAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MaskedConv2d


class GeneralizedPixelVAE(nn.Module):
    """
    PyTorch implementation of the Generalized PixelVAE, adapted for flexibility.

    This module defines the complete VAE, including an encoder that maps
    an input image to a latent distribution, and a conditional PixelCNN decoder
    that reconstructs the image from a latent sample.

    Args:
        input_dim (tuple[int, int, int]): The dimensions of the input images (C, H, W).
        latent_dim (int): Dimensionality of the latent variable z.
        num_pixel_vals (int): Number of possible pixel values (e.g., 256 for 8-bit images).
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        latent_dim: int = 128,
        num_pixel_vals: int = 256,
    ) -> None:
        super(GeneralizedPixelVAE, self).__init__()
        num_channels, height, width = input_dim
        self.input_dim: tuple[int, int, int] = input_dim
        self.latent_dim: int = latent_dim
        self.num_pixel_vals: int = num_pixel_vals

        # ==================
        #      Encoder
        # ==================
        # The original repo uses tf.slim with SAME padding. In PyTorch, a 3x3 kernel
        # with padding=1 gives the same output shape for stride=1. AvgPool2d(2)
        # corresponds to convolutions with stride=2 in the original.
        self.encoder_net: nn.Module = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        # This block programmatically calculates the flattened feature size after the
        # encoder, making the model adaptable to different input resolutions.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            encoder_output_shape = self.encoder_net(dummy_input).shape
            self.encoder_flat_size: int = encoder_output_shape.numel()
            self.decoder_start_shape: tuple[int, ...] = encoder_output_shape[1:]

        # Fully connected layers to output the mean and log-variance of the latent distribution
        self.fc_mu: nn.Module = nn.Linear(self.encoder_flat_size, latent_dim)
        self.fc_log_var: nn.Module = nn.Linear(self.encoder_flat_size, latent_dim)

        # ==================
        #      Decoder
        # ==================
        # The decoder is a conditional PixelCNN. The latent variable z is the condition.
        # This layer projects z to the same spatial feature dimension as the encoder's output.
        self.z_to_decoder: nn.Module = nn.Linear(latent_dim, self.encoder_flat_size)

        # The output of the decoder will have `num_channels * num_pixel_vals` channels
        # to model a categorical distribution over pixel values for each color channel.
        decoder_out_channels: int = num_channels * num_pixel_vals

        self.decoder_net: nn.Module = nn.Sequential(
            # The first masked convolution is type 'A', which blocks connection to the center pixel.
            # The input channels are the image channels + the conditional channels from z.
            MaskedConv2d("A", num_channels + 256, 512, kernel_size=5, padding=2),
            nn.ReLU(True),
            # Subsequent convolutions are type 'B', allowing self-connection.
            MaskedConv2d("B", 512, 512, kernel_size=5, padding=2),
            nn.ReLU(True),
            MaskedConv2d("B", 512, 512, kernel_size=5, padding=2),
            nn.ReLU(True),
            MaskedConv2d("B", 512, decoder_out_channels, kernel_size=5, padding=2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the encoder network to produce the parameters of the approximate
        posterior distribution q(z|x).
        """
        h = self.encoder_net(x).view(x.size(0), -1)  # Flatten the features
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Implements the reparameterization trick to sample from a Gaussian distribution
        in a way that allows gradients to flow back through the sampling process.
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + eps * std

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Runs the decoder network to generate logits for the image reconstruction,
        conditioned on the latent variable z.
        """
        # Project z and reshape it into a spatial feature map to be used as a condition.
        z_cond = self.z_to_decoder(z)
        z_cond = F.relu(z_cond).view(z.size(0), *self.decoder_start_shape)

        # Upsample the condition to match the full image resolution.
        z_cond_upsampled = F.interpolate(
            z_cond, size=(self.input_dim[1], self.input_dim[2]), mode="nearest"
        )

        # Concatenate the upsampled condition with the input image (autoregressive step).
        # During training (teacher forcing), 'x' is the ground truth.
        # During generation, 'x' would be the canvas being drawn upon.
        decoder_input = torch.cat((x, z_cond_upsampled), dim=1)

        # The decoder outputs the logits for the categorical distribution of each pixel value.
        logits = self.decoder_net(decoder_input)
        return logits

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The full forward pass of the VAE.
        1. Encodes the input image `x` into a latent distribution (mu, log_var).
        2. Samples a latent vector `z` using the reparameterization trick.
        3. Decodes `z` to reconstruct the image, using `x` for teacher forcing.
        """
        # Input x is expected to be normalized between [0, 1].
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        # For teacher-forcing, the decoder input should be the ground truth image,
        # typically normalized to [-1, 1] as is common for generative model inputs.
        x_decoder_input = 2 * x - 1

        logits = self.decode(x_decoder_input, z)

        return logits, mu, log_var
