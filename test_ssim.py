from math import isclose
import pytest

import torch

from ssim import compute_ssim

@pytest.fixture
def image1() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.1, 0.1, 0.2, 0.4],
                [0.0, 0.1, 0.4, 0.2],
                [0.5, 0.1, 0.1, 0.1],
                [0.5, 0.8, 0.0, 0.1],
            ],
            [
                [0.1, 0.1, 0.2, 0.4],
                [0.0, 0.1, 0.4, 0.2],
                [0.5, 0.1, 0.1, 0.1],
                [0.5, 0.8, 0.0, 0.1],
            ],
            [
                [0.1, 0.1, 0.2, 0.4],
                [0.0, 0.1, 0.4, 0.2],
                [0.5, 0.1, 0.1, 0.1],
                [0.5, 0.8, 0.0, 0.1],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


@pytest.fixture
def image2() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def test_compute_ssim(
    image1: torch.Tensor,
    image2: torch.Tensor,
) -> None:
    result11 = compute_ssim(image1_t=image1, image2_t=image1)
    result22 = compute_ssim(image1_t=image2, image2_t=image2)
    result12 = compute_ssim(image1_t=image1, image2_t=image2)
    result21 = compute_ssim(image1_t=image2, image2_t=image1)
    assert isclose(result11, 1.0)
    assert isclose(result22, 1.0)
    assert isclose(result12, 0.33096039295196533)
    assert isclose(result21, 0.33096039295196533)
