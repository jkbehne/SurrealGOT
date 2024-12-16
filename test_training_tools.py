import pytest
from pathlib import Path

import torch

from training_tools import (
    SAMPLES_PER_FILE,
    AutoEncoderDataset,
    findPyTorchFiles,
)

def test_findPyTorchFiles_valid_directory(tmp_path: Path):
    pt_file1 = tmp_path / "model1.pt"
    pt_file2 = tmp_path / "model2.pt"
    pt_file1.touch()
    pt_file2.touch()

    found_files = findPyTorchFiles(tmp_path)
    assert len(found_files) == 2
    assert pt_file1 in found_files
    assert pt_file2 in found_files

def test_findPyTorchFiles_invalid_directory():
    with pytest.raises(NotADirectoryError):
        findPyTorchFiles(Path(__file__))

def test_findPyTorchFiles_non_existent_directory():
    with pytest.raises(FileNotFoundError):
        findPyTorchFiles(Path("/non/existent/directory"))

def test_findPyTorchFiles_invalid_input():
    with pytest.raises(TypeError):
        findPyTorchFiles("/invalid/input")

BATCH_SIZE = 32

# Create a temporary directory with some PyTorch files for testing
@pytest.fixture
def tmp_dir(tmp_path: Path):
    for i in range(5):
        pt_file = tmp_path / f"file{i}.pt"
        torch.save(torch.randn(SAMPLES_PER_FILE, 3, 32, 32), pt_file)
    yield tmp_path

def test_autoencoder_dataset_init(tmp_dir: Path):
    dataset = AutoEncoderDataset(base_dir=tmp_dir, batch_size=BATCH_SIZE)
    assert dataset.base_dir == tmp_dir
    assert dataset.batch_size == BATCH_SIZE
    assert len(dataset.files) == 5
    assert dataset.batches_per_file == SAMPLES_PER_FILE // BATCH_SIZE

def test_autoencoder_dataset_len(tmp_dir: Path):
    dataset = AutoEncoderDataset(base_dir=tmp_dir, batch_size=BATCH_SIZE)
    assert len(dataset) == len(dataset.files) * dataset.batches_per_file

def test_autoencoder_dataset_getitem(tmp_dir: Path):
    dataset = AutoEncoderDataset(base_dir=tmp_dir, batch_size=BATCH_SIZE)
    batch, target = dataset[0]
    assert batch.shape == (BATCH_SIZE, 3, 32, 32)
    assert target.shape == (BATCH_SIZE, 3, 32, 32)
    assert torch.all(batch == target)

def test_autoencoder_dataset_reshuffle(tmp_dir: Path):
    dataset = AutoEncoderDataset(base_dir=tmp_dir, batch_size=BATCH_SIZE)
    initial_idxs = dataset.idxs.copy()
    dataset.reshuffle()
    assert dataset.idxs != initial_idxs

def test_autoencoder_dataset_invalid_batch_size(tmp_dir: Path):
    with pytest.raises(ValueError):
        AutoEncoderDataset(base_dir=tmp_dir, batch_size=SAMPLES_PER_FILE + 1)

def test_autoencoder_dataset_invalid_batch_size_divisibility(tmp_dir: Path):
    with pytest.raises(ValueError):
        AutoEncoderDataset(base_dir=tmp_dir, batch_size=11)