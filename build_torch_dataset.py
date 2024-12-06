from pathlib import Path

from dataset_tools import saveAllCrops

def main(
    base_dir: str,
    out_dir: str,
    crop_dim: int = 128,
    batch_size: int = 256,
) -> None:
    """Take a flat folder of png files and save PyTorch crop files

    Take as many crop_dim x crop_dim sized crops of each image in a flat folder
    of png images.

    Args:
        base_dir (str): The folder to grab the png images from
        out_dir (str): The folder to output crops to - made if it doesn't exist
        crop_dim (int): The dimension (square) of the crops. Defaults to 128.

    Raises:
        ValueError: If the base_dir doesn't exist
    """
    base_path = Path(base_dir).expanduser()
    if not base_path.exists():
        raise ValueError("base_dir doesn't exist")
    out_path = Path(out_dir).expanduser()
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    saveAllCrops(
        base_dir=base_path,
        crop_dim=crop_dim,
        out_dir=out_path,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    import fire
    fire.Fire(main)
