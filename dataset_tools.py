from pathlib import Path
from PIL import Image
from random import shuffle

from imageio import imwrite
from imageio.v3 import imread
import numpy as np
import rawpy
import torch
import torchvision.transforms as tforms

from torch_tools import DTYPE

# Determine if we should use the pyheif library instead
HAVE_PYHEIF = False
try:
    import pyheif
    HAVE_PYHEIF = True
except ImportError:
    HAVE_PYHEIF = False

FILE_BATCH_SIZE = 4

def heic2png(heic_path: Path, save_path: Path) -> None:
    """
    Convert HEIC image to PNG.

    Args:
        heic_path (Path): Path to HEIC image.
        save_path (Path): Path to save PNG image.

    Raises:
        ValueError: If the input file does not have a .heic extension.
        FileNotFoundError: If the input file does not exist.
        IsADirectoryError: If the input path is a directory.
        PermissionError: If permission is denied to read the input file or
            write to the output file.
        OSError: If an OS-related error occurs while reading or writing the
            image.

    Notes:
        This function uses the imageio library to read the HEIC image and the
        Pillow library to save it as a PNG.
    """
    heic_name = heic_path.name
    if not heic_name.lower().endswith(".heic"):
        raise ValueError(f"Expected heic file. Got a file named {heic_name}")
    save_name = save_path
    if not save_path.name.lower().endswith(".png"):
        save_name = save_path.with_suffix(".png")
    if HAVE_PYHEIF:
        heif_file = pyheif.read(heic_path.expanduser())
        image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data, "raw",
        )
        image.save(save_name)
    else:
        image_np = imread(heic_path.expanduser())
        image = Image.fromarray(image_np)
        image.save(save_name)

def dng2png(dng_path: Path, save_path: Path) -> None:
    """
    Convert DNG image to PNG.

    Args:
        dng_path (Path): Path to DNG image.
        save_path (Path): Path to save PNG image.

    Raises:
        ValueError: If the input file does not have a .dng extension.
        FileNotFoundError: If the input file does not exist.
        IsADirectoryError: If the input path is a directory.
        PermissionError: If permission is denied to read the input file or
            write to the output file.
        rawpy._rawpy.LibRawIOError: If an error occurs while reading the DNG
            file.
        rawpy._rawpy.LibRawProcessingError: If an error occurs while
            post-processing the DNG file.
        OSError: If an OS-related error occurs while reading or writing the
            image.

    Notes:
        This function uses the rawpy library to read and post-process the DNG
        image, and the imageio library to save it as a PNG.
    """
    lname = dng_path.name.lower()
    if not lname.endswith(".dng"):
        raise ValueError(
            f"Expected dng file. Got a file named {dng_path.name}"
        )
    with rawpy.imread(str(dng_path.expanduser())) as raw:
        rgb = raw.postprocess()
        imwrite(save_path, rgb)

def png2png(png_path: Path, save_path: Path) -> None:
    image = Image.open(png_path.expanduser())
    image.save(save_path)

def png2numpy(png_path: Path) -> np.ndarray:
    image = Image.open(png_path.expanduser())
    return np.array(image)

def _convert2png(in_path: Path, out_path: Path) -> None:
    """
    Convert image file to PNG based on file extension.

    Args:
        in_path (Path): Path to input image file.
        out_path (Path): Path to save output PNG image.

    Notes:
        This function supports conversion of HEIC, DNG, and PNG files.
        If the file extension is not recognized, a message is printed to the
            console.

    Calls:
        heic2png: If the input file has a .heic extension.
        dng2png: If the input file has a .dng extension.
        png2png: If the input file has a .png extension.
    """
    lname = in_path.name.lower()
    if lname.endswith(".heic"):
        heic2png(heic_path=in_path, save_path=out_path)
    elif lname.endswith(".dng"):
        dng2png(dng_path=in_path, save_path=out_path)
    elif lname.endswith(".png"):
        png2png(png_path=in_path, save_path=out_path)
    else:
        print(f"Couldn't recognize extension in file name {in_path.name}")

def convertAllToPNG(base_dir: Path, base_str: str, base_out_dir: Path) -> None:
    """
    Recursively convert all images in a directory to PNG.

    Args:
        base_dir (Path): Base directory to search for images.
        base_str (str): Base string for output file names.
        base_out_dir (Path): Base output directory for PNG images.

    Notes:
        This function recursively traverses the directory tree starting from
            base_dir.
        It converts all supported image files (HEIC, DNG, PNG) to PNG and
            saves them in base_out_dir.
        The output file names are constructed by appending the image index and
            ".png" to base_str.

    Calls:
        _convert2png: To convert individual image files to PNG.
        convertAllToPNG: Recursively, to process subdirectories.
    """
    dir_list: list[Path] = []
    file_list: list[Path] = []
    for path in base_dir.iterdir():
        if path.is_dir():
            dir_list.append(path)
        else:
            file_list.append(path)
    for idx, impath in enumerate(file_list):
        fname = (base_str + "_" + f"image{idx}.png").replace(" ", "")
        _convert2png(impath, base_out_dir / fname)
    for dirpath in dir_list:
        convertAllToPNG(
            base_dir=dirpath,
            base_str=base_str + "_" + dirpath.name,
            base_out_dir=base_out_dir,
        )

def getCropsFromImage(
    image: np.ndarray,
    crop_dim: int,
) -> list[torch.Tensor]:
    if len(image.shape) != 3:
        raise ValueError(
            "Expected image to have 3 dimensions, but got array of shape "
            f"{image.shape}"
        )
    num_rows, num_cols, num_channels = image.shape
    if num_channels != 3:
        raise ValueError(f"Expected 3 channels not {num_channels}")
    num_crop_rows = num_rows // crop_dim
    num_crop_cols = num_cols // crop_dim
    out_list: list[torch.Tensor] = []
    for row_idx in range(num_crop_rows):
        for col_idx in range(num_crop_cols):
            ridx_start = row_idx * crop_dim
            cidx_start = col_idx * crop_dim
            ridx_end = (row_idx + 1) * crop_dim
            cidx_end = (col_idx + 1) * crop_dim
            crop = image[ridx_start:ridx_end, cidx_start:cidx_end, :]
            crop_t = tforms.ToTensor()(crop).to(DTYPE).unsqueeze(0)
            out_list.append(crop_t)
    return out_list

def saveAllCrops(
    base_dir: Path,
    crop_dim: int,
    out_dir: Path,
    batch_size: int,
) -> None:
    if not base_dir.exists():
        raise ValueError(f"Directory {base_dir} doesn't exist")
    paths: list[Path] = []
    for path in base_dir.iterdir():
        if path.is_dir():
            print(
                "Expected a flat directory structure; nested folders won't be "
                "parsed"
            )
        elif not path.name.endswith(".png"):
            print(
                f"Expected to process only png files. Found name {path.name}"
            )
        else: paths.append(path)
    shuffle(paths)
    image_counter = 0
    batch_counter = 0
    t_list: list[torch.Tensor] = []
    for path in paths:
        image_np = png2numpy(png_path=path)
        t_list.extend(getCropsFromImage(image=image_np, crop_dim=crop_dim))
        image_counter += 1
        if (image_counter < FILE_BATCH_SIZE): continue
        else:
            # Batch up the files and reset counter and tensor list
            shuffle(t_list)
            if len(t_list) % batch_size == 0:
                num_batches = len(t_list) // batch_size
            else:
                num_batches = (len(t_list) // batch_size) + 1
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(t_list))
                batch_list = t_list[start_idx:end_idx]
                batch_t = torch.cat(batch_list)
                torch.save(batch_t, out_dir / f"batch{batch_counter}.pt")
                batch_counter += 1
            image_counter = 0
            t_list = []
