import pytest
from pathlib import Path
import torch
import torch.nn as nn
from autoencoder import (
    DCAE,
    saveModel,
    loadModel,
    plotConv2dFilters,
    visualizeConv2dModel,
)

# Define constants
NUM_FILTERS = 16

def test_dcae_init():
    model = DCAE(num_filters=NUM_FILTERS)
    assert isinstance(model, nn.Module)

def test_save_and_load_model(tmp_path: Path):
    model = DCAE(num_filters=NUM_FILTERS)
    save_path = tmp_path / "model.pth"
    saveModel(model, save_path)
    loaded_model = loadModel(save_path)
    assert isinstance(loaded_model, nn.Module)
    assert isinstance(loaded_model, DCAE)

def test_plot_conv2d_filters(tmp_path: Path):
    conv_layer = nn.Conv2d(3, NUM_FILTERS, kernel_size=3)
    save_path = tmp_path / "filters.png"
    plotConv2dFilters(output_path=save_path, conv_layer=conv_layer)
    assert save_path.exists()

def test_visualize_conv2d_model(tmp_path: Path):
    model = DCAE(num_filters=NUM_FILTERS)
    visualizeConv2dModel(model, save_dir=tmp_path)
    assert len(list(tmp_path.glob("filters_layer_*.png"))) == 17
