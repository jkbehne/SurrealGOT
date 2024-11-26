from pathlib import Path
from PIL import Image
import time

from bm3d import bm3d_deblurring, gaussian_kernel
from matplotlib import pyplot as plt
import numpy as np

from denoise_utils import get_experiment_noise

SEED = 0
IMAGE_SCALE = 1.0 / 255.0

def getImage(path: Path) -> np.ndarray:
    return IMAGE_SCALE * np.array(Image.open(path))

def denoiseImage(
    image: np.ndarray,
    noise_type: str,
    noise_var: float,
    kernel_size: int,
    kernel_std: float,
) -> np.ndarray:
    kernel = gaussian_kernel((kernel_size, kernel_size), kernel_std)
    _, psd, _ = get_experiment_noise(
        noise_type=noise_type,
        noise_var=noise_var,
        realization=SEED,
        sz=image.shape,
    )
    return bm3d_deblurring(z=image, sigma_psd=psd, psf=kernel)

def main(
    fpath: str,
    noise_type: str = "g3",
    noise_var: float = 0.01,
    kernel_size: float = 40,
    kernel_std: float = 2.0,
    out_path: str = "images/example.png",
) -> None:
    image = getImage(Path(fpath).expanduser().absolute())
    deblurred = denoiseImage(
        image=image,
        noise_type=noise_type,
        noise_var=noise_var,
        kernel_size=kernel_size,
        kernel_std=kernel_std,
    )
    dbim = Image.fromarray(deblurred)
    dbim.save(out_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
