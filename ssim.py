"""File comment
"""
from typing import Iterable, Optional
import numpy as np
from scipy.signal.windows import gaussian

import torch
from torch.nn.functional import conv2d

from torch_tools import numpy2torch


# Stability constants
C1 = 0.01**2
C2 = 0.03**2


def _create_gaussian_window(size: int, sigma: float, num_channels: int) -> torch.Tensor:
    """Create a Gaussian window trucated to a window size in two dimensions

    Args:
        size (int): The size of the window (produces size x size tensor)
        sigma (float): The width of the (circular) 2D Gaussian window in pixels
        num_channels (int): Number of channels in the image to convolve with

    Returns:
        torch.Tensor: The 2D Gaussian window truncated to the window size
    """
    gauss1d = gaussian(size, sigma)
    gauss1d = np.expand_dims(gauss1d, 1)
    gauss2d = gauss1d @ gauss1d.T
    gauss3d = np.tile(gauss2d, (num_channels, 1, 1, 1))
    return numpy2torch(gauss3d)


def _assert_image_window_match(
    image_shape: Iterable[int],
    window_shape: Iterable[int],
) -> int:
    if len(image_shape) != 4:
        raise TypeError(f"Expected image to be 4D not {len(image_shape)}D")
    if len(window_shape) != 4:
        raise TypeError(f"Expected image to be 4D not {len(window_shape)}D")
    _, num_channels, _, _ = image_shape
    out_channels, in_div_groups, _, _ = window_shape
    if out_channels != num_channels:
        raise ValueError(f"Expected equal in / out channels ({num_channels} != {out_channels})")
    if in_div_groups != 1:
        raise ValueError(f"Expected 2nd dimension of window to be 1 not {in_div_groups}")
    return num_channels


def _get_mu_tensor(
    image: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    num_channels = _assert_image_window_match(image.shape, window.shape)
    return conv2d(image, window, padding="same", groups=num_channels)


def _get_second_moment(
    image: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    return _get_mu_tensor(image=image**2, window=window)


def _get_cross_moment(
    image1: torch.Tensor,
    image2: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    if image1.shape != image2.shape:
        raise ValueError(
            f"image1 and image2 must have the same shape ({image1.shape} != {image2.shape})"
        )
    return _get_mu_tensor(image=image1 * image2, window=window)


def _get_num_channels(image_shape: Iterable[int]) -> int:
    if len(image_shape) != 4:
        raise TypeError(f"Expected image to be 4D not {len(image_shape)}D")
    _, num_chnanels, _, _ = image_shape
    return num_chnanels


def compute_ssim(
    image1_t: torch.Tensor,
    image2_t: torch.Tensor,
    window: Optional[torch.Tensor] = None,
) -> float:
    if window is None:
        num_channels = _get_num_channels(image1_t.shape)
        window = _create_gaussian_window(11, 1.5, num_channels)
    mu1 = _get_mu_tensor(image=image1_t, window=window)
    mu2 = _get_mu_tensor(image=image2_t, window=window)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu_cross = mu1 * mu2

    var1 = _get_second_moment(image=image1_t, window=window) - mu1_sq
    var2 = _get_second_moment(image=image2_t, window=window) - mu2_sq
    cov = _get_cross_moment(image1=image1_t, image2=image2_t, window=window) - mu_cross

    num = (2.0 * mu_cross + C1) * (2.0 * cov + C2)
    den = (mu1_sq + mu2_sq + C1) * (var1 + var2 + C2)
    ssim = torch.mean(num / den)
    return 1.0 - torch.maximum(ssim, torch.zeros_like(ssim))
