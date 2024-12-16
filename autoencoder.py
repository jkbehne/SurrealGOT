from math import ceil, sqrt
from pathlib import Path

from matplotlib import pyplot as plt
import torch
import torch.nn as nn

class DCAE(nn.Module):
    def __init__(self, num_filters: int = 64):
        super(DCAE, self).__init__()
        layers = [
            nn.Sequential(  # Layer 1
                nn.Conv2d(
                    3,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=7,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 2
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=7,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 3
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=7,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 4
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=5,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 5
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=5,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 6
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=5,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 7
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=5,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 8
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 9
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 10
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 11
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 12
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 13
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 14
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 15
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 16
                nn.Conv2d(
                    num_filters,  # Input channels
                    num_filters,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Layer 17
                nn.Conv2d(
                    num_filters,  # Input channels
                    3,  # Output channels
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            ),
        ]
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        return self.layers(inputs)

def saveModel(model: nn.Module, savePath: Path) -> None:
    torch.save(model, savePath)

def loadModel(loadPath: Path) -> nn.Module:
    return torch.load(loadPath)

def plotConv2dFilters(output_path: Path, conv_layer: nn.Conv2d) -> None:
    """
    Plot filters in the given 2D convolutional layer. Saves to output_path.

    Args:
        output_path (Path): The path to save the plot to.
        conv_layer (nn.Conv2d): The 2D convolutional layer.
    """
    # Get the weights of the convolutional layer
    weights = conv_layer.weight.detach().cpu().numpy()

    # Get the number of filters and the kernel size
    num_filters = weights.shape[0]

    # Create a figure with a grid of subplots
    num_rows = int(ceil(sqrt(num_filters)))
    _, axs = plt.subplots(nrows=num_rows, ncols=num_rows, figsize=(10, 10))
    axs = axs.flatten()

    # Iterate over the filters and plot them
    for i in range(num_filters):
        axs[i].imshow(weights[i, 0, :, :], cmap="gray")

    # Hide any unused subplots
    for i in range(num_filters, len(axs)):
        axs[i].axis("off")

    # Layout so plots do not overlap
    plt.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)

    plt.close()

def visualizeConv2dModel(
    model: torch.nn.Module,
    save_dir: Path,
) -> None:
    """
    Visualize the weights of the convolutional layers in the given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        save_dir (Path): The base directory to save layer plots to.
    """
    # Get the convolutional layers from the model
    conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]

    for idx, layer in enumerate(conv_layers):
        save_path = save_dir / f"filters_layer_{idx}.png"
        plotConv2dFilters(output_path=save_path, conv_layer=layer)
