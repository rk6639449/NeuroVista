"""
Integration tests for Neurite compatibility.

Tests critical Neurite components that VoxelMorph depends on:
- BasicUNet (model backbone)
- volshape_to_ndgrid (grid generation)
- ConvBlock (flow layer)
"""

import pytest
import torch
import neurite as ne
import neurite.nn.functional as nef


@pytest.fixture
def shape_2d():
    """2D spatial shape for testing."""
    return (64, 64)


@pytest.fixture
def shape_3d():
    """3D spatial shape for testing."""
    return (32, 32, 32)


@pytest.fixture
def sample_input_2d(shape_2d):
    """Sample 2D input tensor."""
    torch.manual_seed(42)
    return torch.randn(1, 1, *shape_2d)


@pytest.fixture
def sample_input_3d(shape_3d):
    """Sample 3D input tensor."""
    torch.manual_seed(42)
    return torch.randn(1, 1, *shape_3d)


def test_basic_unet_forward_2d(sample_input_2d):
    """Test BasicUNet produces expected output shape for 2D."""
    model = ne.nn.models.BasicUNet(
        ndim=2,
        in_channels=1,
        out_channels=2,
        nb_features=[8, 8, 8, 8]
    )
    output = model(sample_input_2d)

    assert output.shape[0] == sample_input_2d.shape[0]
    assert output.shape[1] == 2
    assert output.shape[-2:] == sample_input_2d.shape[-2:]


def test_basic_unet_forward_3d(sample_input_3d):
    """Test BasicUNet produces expected output shape for 3D."""
    model = ne.nn.models.BasicUNet(
        ndim=3,
        in_channels=1,
        out_channels=3,
        nb_features=[8, 8, 8, 8]
    )
    output = model(sample_input_3d)

    assert output.shape[0] == sample_input_3d.shape[0]
    assert output.shape[1] == 3
    assert output.shape[-3:] == sample_input_3d.shape[-3:]


def test_basic_unet_gradient_flow(sample_input_3d):
    """Test gradients flow through BasicUNet."""
    model = ne.nn.models.BasicUNet(
        ndim=3,
        in_channels=1,
        out_channels=3,
        nb_features=[8, 8, 8, 8]
    )
    sample_input_3d.requires_grad_(True)
    output = model(sample_input_3d)
    loss = output.sum()
    loss.backward()

    assert sample_input_3d.grad is not None
    assert sample_input_3d.grad.shape == sample_input_3d.shape


def test_volshape_to_ndgrid_ij_indexing_2d():
    """
    Test volshape_to_ndgrid produces grids with ij indexing.

    With (ndim, *spatial) format, grid[0] is the first coordinate channel (row index),
    and grid[1] is the second coordinate channel (col index).
    """
    shape = (64, 64)
    grid = ne.volshape_to_ndgrid(shape, indexing="ij", stack=True)

    assert grid.shape == (2, *shape)

    # ij indexing: first coord varies with first spatial index (row)
    assert grid[0, 0, 0] < grid[0, 1, 0]      # first coord increases with i (row)
    assert grid[0, 0, 0] == grid[0, 0, 1]     # first coord constant along j (col)
    assert grid[1, 0, 0] == grid[1, 1, 0]     # second coord constant along i (row)
    assert grid[1, 0, 0] < grid[1, 0, 1]      # second coord increases with j (col)


def test_volshape_to_ndgrid_xy_indexing_2d():
    """
    Test volshape_to_ndgrid produces grids with xy indexing.

    With (ndim, *spatial) format and xy indexing, grid[0] is x (varies with col),
    and grid[1] is y (varies with row).
    """
    shape = (64, 64)
    grid = ne.volshape_to_ndgrid(shape, indexing="xy", stack=True)

    assert grid.shape == (2, *shape)

    # xy indexing: first coord (x) varies with second spatial index (columns)
    assert grid[0, 0, 0] == grid[0, 1, 0]     # x constant along rows (i)
    assert grid[0, 0, 0] < grid[0, 0, 1]      # x increases along columns (j)
    assert grid[1, 0, 0] < grid[1, 1, 0]      # y increases along rows (i)
    assert grid[1, 0, 0] == grid[1, 0, 1]     # y constant along columns (j)


def test_volshape_to_ndgrid_ij_indexing_3d():
    """
    Test volshape_to_ndgrid produces grids with ij indexing for 3D.

    With (ndim, *spatial) format, grid[d] is the coordinate channel for dimension d.
    """
    shape = (32, 32, 32)
    grid = ne.volshape_to_ndgrid(shape, indexing="ij", stack=True)

    assert grid.shape == (3, *shape)

    # ij indexing: coords align with indices
    assert grid[0, 0, 0, 0] < grid[0, 1, 0, 0]  # coord 0 increases with dim 0
    assert grid[1, 0, 0, 0] < grid[1, 0, 1, 0]  # coord 1 increases with dim 1
    assert grid[2, 0, 0, 0] < grid[2, 0, 0, 1]  # coord 2 increases with dim 2


def test_volshape_to_ndgrid_xy_indexing_3d():
    """
    Test volshape_to_ndgrid produces grids with xy indexing for 3D.

    With (ndim, *spatial) format and xy indexing, coordinates are reordered.
    """
    shape = (32, 32, 32)
    grid = ne.volshape_to_ndgrid(shape, indexing="xy", stack=True)

    assert grid.shape == (3, *shape)

    # xy indexing: coords are reordered relative to spatial dims
    assert grid[0, 0, 0, 0] == grid[0, 1, 0, 0]  # x constant along dim 0
    assert grid[0, 0, 0, 0] < grid[0, 0, 1, 0]   # x increases with dim 1
    assert grid[1, 0, 0, 0] < grid[1, 1, 0, 0]   # y increases with dim 0


def test_conv_block_forward_2d(shape_2d):
    """Test ConvBlock forward pass for 2D."""
    torch.manual_seed(42)
    block = ne.nn.modules.ConvBlock(ndim=2, in_channels=8, out_channels=2)
    input_tensor = torch.randn(1, 8, *shape_2d)
    output = block(input_tensor)

    assert output.shape == (1, 2, *shape_2d)
    assert output.requires_grad is True


def test_conv_block_forward_3d(shape_3d):
    """Test ConvBlock forward pass for 3D."""
    torch.manual_seed(42)
    block = ne.nn.modules.ConvBlock(ndim=3, in_channels=8, out_channels=3)
    input_tensor = torch.randn(1, 8, *shape_3d)
    output = block(input_tensor)

    assert output.shape == (1, 3, *shape_3d)
    assert output.requires_grad is True
