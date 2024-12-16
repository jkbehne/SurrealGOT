"""
"""
import json
from pathlib import Path
from random import shuffle
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Number of samples per file is assumed to be 256, but that could change
SAMPLES_PER_FILE = 256

def findPyTorchFiles(base_dir: Path) -> list[Path]:
    """
    Recursively finds all .pt files in the given base directory.

    Args:
        base_dir (Path): The base directory to search for .pt files.

    Returns:
        list[Path]: A list of Path objects representing the found .pt files.

    Raises:
        TypeError: If base_dir is not a pathlib.Path object.
        FileNotFoundError: If base_dir does not exist.
        NotADirectoryError: If base_dir is not a directory.
    """
    # Check if base_dir is a Path object
    if not isinstance(base_dir, Path):
        raise TypeError("base_dir must be a pathlib.Path object")

    # Check if base_dir exists
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir '{base_dir}' does not exist")

    # Check if base_dir is a directory
    if not base_dir.is_dir():
        raise NotADirectoryError(f"base_dir '{base_dir}' is not a directory")

    # Recursively find all .pt files
    return list(base_dir.rglob("*.pt"))

class AutoEncoderDataset(Dataset):
    def __init__(self, base_dir: Path, batch_size: int):
        # Check if we can work with the batch size parameter
        if batch_size > SAMPLES_PER_FILE:
            raise ValueError(
                "batch_size can't be greater than samples per file"
            )
        if SAMPLES_PER_FILE % batch_size != 0:
            raise ValueError(
                "batch_size must divide evenly into samples per file"
            )

        # Set passed in parameters
        self.batch_size = batch_size
        self.base_dir = base_dir.expanduser()

        # Get the PyTorch files in the directory
        self.files = findPyTorchFiles(base_dir=self.base_dir)

        # Work out the initial idxs
        self.idxs = list(range(len(self.files)))
        shuffle(self.idxs)

        # Determine batches per file and set initial index
        self.batches_per_file = SAMPLES_PER_FILE // batch_size

        # Initialize the current file data to None
        self.current_file_data: Optional[torch.Tensor] = None

        # Initialize the last batch idx to zero
        self.current_batch_idx = 0

    def reshuffle(self) -> None:
        shuffle(self.idxs)

    def __len__(self) -> int:
        return self.batches_per_file * len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.current_batch_idx = index
        # Determine the index split
        file_idx = index // self.batches_per_file
        batch_idx = index % self.batches_per_file
        if batch_idx == 0:
            # Open up a new file
            fpath = self.files[file_idx]
            print(f"Loading file {fpath}")
            self.current_file_data = torch.load(fpath)
            print("Finished loading file")

            # Check if we have a full file first
            num_samples = self.current_file_data.shape[0]
            if num_samples < SAMPLES_PER_FILE:
                print(f"File had {num_samples} samples, expected {SAMPLES_PER_FILE}")

                # Make a new tensor that has the full data
                new_idxs = np.random.choice(
                    num_samples, size=SAMPLES_PER_FILE, replace=True,
                ).tolist()
                self.current_file_data = self.current_file_data[new_idxs, :, :, :]

        start_idx = batch_idx * self.batch_size
        end_idx = (batch_idx + 1) * self.batch_size
        batch = self.current_file_data[start_idx:end_idx, :, :, :]
        return batch, batch  # Target is the input in an autoencoder

    def save(self, save_path: Path) -> None:
        data = {
            "base_dir": str(self.base_dir),
            "batch_size": self.batch_size,
            "current_batch_idx": self.current_batch_idx,
            "file_idxs": self.idxs,
        }
        with open(save_path, "w") as ofile:
            json.dump(data, ofile)
