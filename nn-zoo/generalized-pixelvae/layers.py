import torch
import torch.nn as nn
from typing import Tuple, Union


class MaskedConv2d(nn.Conv2d):
    """
    A custom 2D convolution layer with masking for autoregressive models.

    Mask
            -------------------------------------
           |  1       1       1       1       1 |
           |  1       1       1       1       1 |
           |  1       1    1 if B     0       0 |   H // 2
           |  0       0       0       0       0 |   H // 2 + 1
           |  0       0       0       0       0 |
            -------------------------------------
    index     0       1     W//2    W//2+1

    Args:
        mask_type (str): 'A' or 'B'.
            - 'A': Used for the first layer, masks the center pixel.
            - 'B': Used for subsequent layers, allows connection to the center pixel.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): Padding added to the input.
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )

        if mask_type not in ["A", "B"]:
            raise ValueError("Invalid mask_type. Must be 'A' or 'B'.")
        self.mask_type: str = mask_type

        # Create and register the mask as a buffer, so it's not a trainable parameter
        mask = self.weight.data.clone()
        mask.fill_(1)

        # Using type assertion for clarity as '.size()' returns torch.Size which is a tuple
        _, _, kH, kW = self.weight.size()

        mask[:, :, kH // 2, kW // 2 + 1 :] = 0
        mask[:, :, kH // 2 + 1 :, :] = 0

        if self.mask_type == "A":
            mask[:, :, kH // 2, kW // 2] = 0

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the mask to the weights before convolution
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
