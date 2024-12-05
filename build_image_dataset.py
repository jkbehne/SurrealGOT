from pathlib import Path

from dataset_tools import convertAllToPNG

def main(path_list: list[str], base_list: list[str], out_dir: str) -> None:
    """Main function for building a dataset of png images

    Basic idea is that we recursively search through each directory given in
    the path_list variable and turn every image found within those directories
    into a png file (heic, png, and dng inputs are currently supported). We 
    write each corresponding png file into a single directory, given by the
    out_dir variable. The name of each file will be as follows:

        out_name = base_dir_next_dir_...final_dir_image{idx}

    where idx is the image number within the folder that the image was found.

    Raises:
        ValueError: If path_list and base_list aren't the same length

    Args:
        path_list: List of paths to search for images
        base_list: List of name abbreviations for each path
        out_dir: The directory to output the png files to
    """
    if len(path_list) != len(base_list):
        raise ValueError("path_list and base_list must be the same length")
    out_path = Path(out_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    for path_str, base_str in zip(path_list, base_list):
        convertAllToPNG(
            base_dir=Path(path_str),
            base_str=base_str,
            base_out_dir=out_path,
        )

if __name__ == "__main__":
    import fire
    fire.Fire(main)
