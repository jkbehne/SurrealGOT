"""
The intent of this file is to house functions needed to create a dataset for training
on subimages of larger, high-resolution images. The main goal is to random sample cropped
versions of the larger images and ultimate write those to a PyTorch tensor (.pt) file.
"""
from pathlib import Path
from typing import List

from imageio.v3 import imread
import numpy as np
from numpy.typing import NDArray
import torch
import torchvision.transforms as tforms


def get_paths(path: Path) -> List[Path]:
    output: List[Path] = []
    if not path.is_dir():
        raise ValueError(f"Expected only directories not file {path}")
    for file_or_dir in path.iterdir():
        if file_or_dir.is_dir(): output.extend(get_paths(file_or_dir))
        if str(file_or_dir).lower().endswith(".heic"): output.append(file_or_dir)
    return output


def get_image_crops(
    image: NDArray[np.float64],
    crop_size: int,
    num_crops: int,
) -> List[torch.Tensor]:
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected image to have 3-dimensions but got array of shape {image.shape}"
        )
    num_rows, num_cols, num_channels = image.shape
    if num_channels != 3:
        raise ValueError(f"Expected image to have 3 channels not {num_channels}")
    max_row_idx = num_rows - crop_size
    max_col_idx = num_cols - crop_size
    crop_list: List[torch.Tensor] = []
    for _ in range(num_crops):
        start_row_idx = np.random.choice(max_row_idx)
        end_row_idx = start_row_idx + crop_size
        start_col_idx = np.random.choice(max_col_idx)
        end_col_idx = start_col_idx + crop_size
        crop = image[start_row_idx:end_row_idx, start_col_idx:end_col_idx, :]
        crop_list.append(tforms.ToTensor()(crop).unsqueeze(0))
    return crop_list


def get_crop_dataset(
    paths: List[Path],
    crop_size: int,
    num_crops_per_image: int,
    verbose: bool,
) -> torch.Tensor:
    crop_list: List[torch.Tensor] = []
    for idx, path in enumerate(paths):
        if verbose:
            print(f"Getting sub-images for sample {idx + 1} of {len(paths)}")
        crop_list.extend(
            get_image_crops(image=imread(path), crop_size=crop_size, num_crops=num_crops_per_image)
        )
    return torch.cat(crop_list)


def main(
    base_dir: str = "~/code/SurrealGOT/images/EWITW",
    crop_size: int = 128,
    num_crops_per_image: int = 42,
    out_path: str = "~/code/SurrealGOT/datasets/EWITW.pt",
    verbose: bool = True,
) -> None:
    paths = get_paths(Path(base_dir).expanduser())
    ds_tensor = get_crop_dataset(
        paths=paths,
        crop_size=crop_size,
        num_crops_per_image=num_crops_per_image,
        verbose=verbose,
    )
    if verbose:
        print(f"Final tensor shape = {ds_tensor.shape}")
    torch.save(ds_tensor, Path(out_path).expanduser())


if __name__ == "__main__":
    import fire; fire.Fire(main)
