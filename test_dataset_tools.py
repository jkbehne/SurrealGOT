import pytest
from pathlib import Path
from PIL import Image
import shutil

from imageio.v3 import imwrite
import numpy as np
import torch

from dataset_tools import (
    convertAllToPNG,
    dng2png,
    heic2png,
    png2png,
    png2numpy,
    saveAllCrops,
    saveCropsFromImage,
)


def create_fake_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)


def test_heic2png(tmp_path: Path):
    # Create a temporary HEIC file
    heic_path = tmp_path / "image.heic"
    image = create_fake_image()
    imwrite(heic_path, image)

    # Convert HEIC to PNG
    png_path = tmp_path / "image.png"
    heic2png(heic_path, png_path)

    # Check if PNG file exists
    assert png_path.exists()

    # Check if PNG file is not empty
    assert png_path.stat().st_size > 0


def test_heic2png_invalid_extension(tmp_path: Path):
    # Create a temporary file with an invalid extension
    invalid_path = tmp_path / "image.invalid"

    # Try to convert invalid file to PNG
    png_path = tmp_path / "image.png"
    with pytest.raises(ValueError):
        heic2png(invalid_path, png_path)


def test_heic2png_non_existent_file(tmp_path: Path):
    # Try to convert non-existent file to PNG
    non_existent_path = tmp_path / "non_existent.heic"
    png_path = tmp_path / "image.png"
    with pytest.raises(FileNotFoundError):
        heic2png(non_existent_path, png_path)


def test_dng2png(tmp_path: Path):
    dng_path = Path("test.dng")
    # Note: Creating a valid DNG file is complex and requires a library like rawpy.
    # For this test, we'll assume that the DNG file already exists.
    # If you want to generate a DNG file, you'll need to use a library like rawpy.

    # Convert DNG to PNG
    png_path = tmp_path / "image.png"
    dng2png(dng_path, png_path)

    # Check if PNG file exists
    assert png_path.exists()

    # Check if PNG file is not empty
    assert png_path.stat().st_size > 0


def test_dng2png_invalid_extension(tmp_path: Path):
    # Create a temporary file with an invalid extension
    invalid_path = tmp_path / "image.invalid"

    # Try to convert invalid file to PNG
    png_path = tmp_path / "image.png"
    with pytest.raises(ValueError):
        dng2png(invalid_path, png_path)


def test_png2png(tmp_path: Path):
    # Create a temporary PNG file
    png_path = tmp_path / "image.png"
    image = Image.new('RGB', (100, 100))
    image.save(png_path)

    # Copy PNG file
    new_png_path = tmp_path / "new_image.png"
    png2png(png_path, new_png_path)

    # Check if new PNG file exists
    assert new_png_path.exists()

    # Check if new PNG file is not empty
    assert new_png_path.stat().st_size > 0


def test_png2png_non_existent_file(tmp_path: Path):
    # Try to copy non-existent PNG file
    non_existent_path = tmp_path / "non_existent.png"
    new_png_path = tmp_path / "new_image.png"
    with pytest.raises(FileNotFoundError):
        png2png(non_existent_path, new_png_path)


def test_png2numpy(tmp_path: Path):
    # Create a temporary PNG file
    png_path = tmp_path / "image.png"
    image = Image.new('RGB', (100, 100))
    image.save(png_path)

    # Convert PNG to NumPy array
    image_np = png2numpy(png_path)

    # Check if NumPy array has correct shape
    assert image_np.shape == (100, 100, 3)


def test_png2numpy_non_existent_file(tmp_path: Path):
    # Try to convert non-existent PNG file to NumPy array
    non_existent_path = tmp_path / "non_existent.png"
    with pytest.raises(FileNotFoundError):
        png2numpy(non_existent_path)

def test_convertAllToPNG(tmp_path: Path):
    # Create a temporary directory with subdirectories and image files
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    sub_dir = base_dir / "subdir"
    sub_dir.mkdir()
    heic_path = base_dir / "image.heic"
    dng_path = base_dir / "image.dng"
    png_path = base_dir / "image.png"
    sub_heic_path = sub_dir / "sub_image.heic"
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    imwrite(heic_path, image)
    shutil.copy("test.DNG", dng_path)
    # imwrite(dng_path, image)
    imwrite(png_path, image)
    imwrite(sub_heic_path, image)

    # Convert all images to PNG
    base_out_dir = tmp_path / "output"
    base_out_dir.mkdir()
    convertAllToPNG(base_dir, "images", base_out_dir)

    # Check if PNG files exist
    assert (base_out_dir / "images_image0.png").exists()
    assert (base_out_dir / "images_image1.png").exists()
    assert (base_out_dir / "images_image2.png").exists()
    assert (base_out_dir / "images_subdir_image0.png").exists()


def test_convertAllToPNG_empty_directory(tmp_path: Path):
    # Create an empty temporary directory
    base_dir = tmp_path / "images"
    base_dir.mkdir()

    # Convert all images to PNG
    base_out_dir = tmp_path / "output"
    base_out_dir.mkdir()
    convertAllToPNG(base_dir, "image", base_out_dir)

    # Check if no PNG files were created
    assert not any(file.name.endswith(".png") for file in base_out_dir.iterdir())


def test_convertAllToPNG_non_existent_directory(tmp_path: Path):
    # Try to convert all images in a non-existent directory
    base_dir = tmp_path / "non_existent"
    base_out_dir = tmp_path / "output"
    base_out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        convertAllToPNG(base_dir, "image", base_out_dir)


def test_saveCropsFromImage(tmp_path: Path):
    # Create a temporary directory
    base_dir = tmp_path

    # Create a sample image
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # Save crops from the image
    crop_dim = 50
    counter = 0
    updated_counter = saveCropsFromImage(image, crop_dim, counter, base_dir)

    # Check if the correct number of crops were saved
    assert updated_counter == 4

    # Check if each crop file exists and can be loaded
    for i in range(updated_counter):
        fname = base_dir / f"sample_{i}.pt"
        assert fname.exists()
        crop_t = torch.load(fname)
        assert crop_t.shape == (3, crop_dim, crop_dim)


def test_saveCropsFromImage_invalid_image_shape(tmp_path: Path):
    # Create a temporary directory
    base_dir = tmp_path

    # Create a sample image with invalid shape
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # Try to save crops from the image
    crop_dim = 50
    counter = 0
    with pytest.raises(ValueError):
        saveCropsFromImage(image, crop_dim, counter, base_dir)


def test_saveCropsFromImage_invalid_image_channels(tmp_path: Path):
    # Create a temporary directory
    base_dir = tmp_path

    # Create a sample image with invalid number of channels
    image = np.random.randint(0, 256, size=(100, 100, 4), dtype=np.uint8)

    # Try to save crops from the image
    crop_dim = 50
    counter = 0
    with pytest.raises(ValueError):
        saveCropsFromImage(image, crop_dim, counter, base_dir)


def test_saveAllCrops(tmp_path: Path):
    # Create a temporary directory with PNG images
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    png_path1 = base_dir / "image1.png"
    png_path2 = base_dir / "image2.png"
    image1 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    imwrite(png_path1, image1)
    imwrite(png_path2, image2)

    # Create an output directory
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Save crops from all PNG images
    crop_dim = 50
    saveAllCrops(base_dir, crop_dim, out_dir)

    # Check if the correct number of crops were saved
    assert len(list(out_dir.glob("*.pt"))) == 8


def test_saveAllCrops_empty_directory(tmp_path: Path):
    # Create an empty temporary directory
    base_dir = tmp_path / "images"
    base_dir.mkdir()

    # Create an output directory
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Save crops from all PNG images (should be none)
    crop_dim = 50
    saveAllCrops(base_dir, crop_dim, out_dir)

    # Check if no crops were saved
    assert len(list(out_dir.glob("*.pt"))) == 0


def test_saveAllCrops_non_existent_directory(tmp_path: Path):
    # Try to save crops from a non-existent directory
    base_dir = tmp_path / "non_existent"
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    crop_dim = 50
    with pytest.raises(ValueError):
        saveAllCrops(base_dir, crop_dim, out_dir)

    # Check if no crops were saved
    assert len(list(out_dir.glob("*.pt"))) == 0


def test_saveAllCrops_nested_directory(tmp_path: Path):
    # Create a temporary directory with a nested directory
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    nested_dir = base_dir / "nested"
    nested_dir.mkdir()
    png_path = nested_dir / "image.png"
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    imwrite(png_path, image)

    # Create an output directory
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Save crops from all PNG images (should skip nested directory)
    crop_dim = 50
    saveAllCrops(base_dir, crop_dim, out_dir)

    # Check if no crops were saved
    assert len(list(out_dir.glob("*.pt"))) == 0


def test_saveAllCrops_non_png_file(tmp_path: Path):
    # Create a temporary directory with a non-PNG file
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    non_png_path = base_dir / "image.jpg"
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    imwrite(non_png_path, image)

    # Create an output directory
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Save crops from all PNG images (should skip non-PNG file)
    crop_dim = 50
    saveAllCrops(base_dir, crop_dim, out_dir)

    # Check if no crops were saved
    assert len(list(out_dir.glob("*.pt"))) == 0