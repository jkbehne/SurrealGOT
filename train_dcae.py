from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional

import torch
from torchvision import transforms
from torchvision.utils import save_image

from autoencoder import (
    DCAE,
    loadModel,
    saveModel,
    visualizeConv2dModel,
)
from ssim import compute_ssim
from torch_tools import DEVICE, DTYPE
from training_tools import AutoEncoderDataset

def _pngDirToTensor(directory: Path) -> tuple[list[torch.Tensor], list[Path]]:
    """
    Load all PNG images in the given directory into a list of tensors.

    Args:
        directory (Path): The directory to load PNG images from.

    Returns:
        tuple[torch.Tensor, list[Path]]: Output tensor & list of output paths.
    """
    image_tensors: list[torch.Tensor] = []
    png_paths: list[Path] = []
    for file in directory.iterdir():
        if file.suffix.lower() == ".png":
            image = Image.open(file)

            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor: torch.Tensor = transform(image)
            image_tensors.append(image_tensor.to(device=DEVICE, dtype=DTYPE))

            ofile = file.parent / (file.stem + "_out.png")
            png_paths.append(ofile)
    return image_tensors, png_paths

class DCAETrainer:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        batch_size: int,
        learn_rate: float,
        num_epochs: int,
        save_rate_batches: int,
        num_filters: int,
        test_image_dir: Optional[Path],
    ):
        # Setup the output save directory
        date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.output_path = output_dir / f"dcae_run_{date_str}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup the dataset
        self.dataset = AutoEncoderDataset(
            base_dir=dataset_dir,
            batch_size=batch_size,
        )

        # Save the training parameters
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs

        # Setup the save rate in terms of batches
        self.save_rate = save_rate_batches

        # Setup the DCAE model
        self.model = DCAE(num_filters=num_filters).to(device=DEVICE, dtype=DTYPE)

        # Pre-load the test images if a path was given
        if test_image_dir is not None:
            self.test_images, self.test_out_paths = _pngDirToTensor(
                directory=test_image_dir,
            )
        else: self.test_images: Optional[list[torch.Tensor]] = None

    def run(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

        for epoch in range(self.num_epochs):
            print(f"Running training epoch {epoch}")
            if epoch != 0: self.dataset.reshuffle()
            batch_counter = 0
            for (input, target) in self.dataset:
                # Do the forward pass
                output = self.model(input.to(device=DEVICE, dtype=DTYPE))
                loss: torch.Tensor = compute_ssim(
                    image1_t=target.to(device=DEVICE, dtype=DTYPE),
                    image2_t=output,
                )

                # Do the optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_counter >= self.save_rate:
                    print(f"Current loss = {loss.item()}")
                    self.save()
                batch_counter += 1

    def save(self) -> None:
        # Save states in case we need to reload
        self.dataset.save(save_path=self.output_path / "ds_state.json")
        saveModel(model=self.model, savePath=self.output_path / "model_state.pth")

        # Visualize the current state of the model
        visualizeConv2dModel(model=self.model, save_dir=self.output_path)

        # Run the model on test images if we have them
        if self.test_images is not None:
            with torch.no_grad():
                for image_t, opath in zip(self.test_images, self.test_out_paths):
                    out_t: torch.Tensor = self.model(image_t.unsqueeze(0))
                    out_t = out_t.squeeze(0).cpu().detach()
                    save_image(out_t, opath)

def main(
    dataset_dir: str,
    output_dir: str,
    batch_size: int = 32,
    learn_rate: float = 0.001,
    num_epochs: int = 80,
    save_rate_batches: int = 128,
    num_filters: int = 64,
    test_image_dir: Optional[str] = None,
) -> None:
    test_dir = (
        Path(test_image_dir).expanduser() if test_image_dir is not None else None
    )
    trainer = DCAETrainer(
        dataset_dir=Path(dataset_dir).expanduser(),
        output_dir=Path(output_dir).expanduser(),
        batch_size=batch_size,
        learn_rate=learn_rate,
        num_epochs=num_epochs,
        save_rate_batches=save_rate_batches,
        num_filters=num_filters,
        test_image_dir=test_dir,
    )
    trainer.run()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
