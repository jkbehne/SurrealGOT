import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from torch_tools import (
    DEVICE,
    DTYPE,
    numpy2torch,
    torch2numpy,
    load_heic_image,
)


@pytest.fixture
def simple_numpy() -> NDArray[np.float64]:
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


@pytest.fixture
def simple_torch() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)


@pytest.mark.parametrize("copy", [False, True])
def test_numpy2torch(
    simple_numpy: NDArray[np.float64],
    simple_torch: torch.Tensor,
    copy: bool,
) -> None:
    # Test that round trip conversion works as expected
    result_t = numpy2torch(simple_numpy, copy=copy)
    assert torch.allclose(result_t, simple_torch)
    assert result_t.device == DEVICE
    assert result_t.dtype == DTYPE


def test_torch2numpy(
    simple_numpy: NDArray[np.float64],
    simple_torch: torch.Tensor,
) -> None:
    result_np = torch2numpy(simple_torch)
    assert np.allclose(result_np, simple_numpy)
    assert result_np.dtype == np.float64


def test_load_heic_image() -> None:
    result = load_heic_image(fname="test.HEIC")
    assert result.shape == torch.Size([1, 3, 3024, 4032])
