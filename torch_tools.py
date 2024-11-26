"""
The intent of this file is to provide basic mechanisms for interfacing with PyTorch in both a CPU
and GPU based environment. Note: CUDA availability is automatically detected and conversions
default to using CUDA if it's available. The default data type is 64-bit floating point (though
this can be changed in the transformations as needed).
"""
from pathlib import Path

from imageio.v3 import imread
import numpy as np
from numpy.typing import NDArray
import torch
import torchvision.transforms as tforms


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")
DTYPE = torch.float32


def numpy2torch(
    x: NDArray[np.float64],
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    copy: bool = False,
) -> torch.Tensor:
    return (torch.from_numpy(x)).to(device=device, dtype=dtype, copy=copy)


def torch2numpy(
    x: torch.Tensor,
    dtype: np.dtype = np.float64,
) -> NDArray[np.float64]:
    return x.cpu().detach().numpy().astype(dtype)


def load_heic_image(fname: str) -> torch.Tensor:
    fpath = Path(fname)
    efpath = fpath.expanduser().absolute()
    if not str(efpath).lower().endswith(".heic"):
        raise ValueError("Expected fpath to end with .heic")
    image = imread(efpath)
    return tforms.ToTensor()(image).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
