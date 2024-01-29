import torch
import torch.nn as nn
import numpy as np


class ConvWithNorms(nn.Module):

    def __init__(self, num_channels: int, kernel_size: int, stride: int,
                 padding: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, stride,
                              padding)
        self.batchnorm = nn.BatchNorm2d(num_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.gelu(batchnorm_res)


class ConvTransposeWithNorms(nn.Module):

    def __init__(self,
                 num_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int = 0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(num_channels,
                                       num_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       output_padding=output_padding)
        self.batchnorm = nn.BatchNorm2d(num_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.gelu(batchnorm_res)


class StepDown(nn.Module):

    def __init__(self, num_channels: int, num_filters_per_block: int):
        super().__init__()
        # Single stride 2 conv layer, followed by num_filters stride 1 conv layers
        # Each conv is followed by a batchnorm and a gelu
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvWithNorms(num_channels, 3, 2, 1))
        for _ in range(num_filters_per_block):
            self.conv_layers.append(ConvWithNorms(num_channels, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x


class StepUp(nn.Module):

    def __init__(self, num_channels: int, num_filters_per_block: int):
        super().__init__()
        # Single stride 2 conv layer, followed by num_filters stride 1 conv layers
        # Each conv is followed by a batchnorm and a gelu
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            ConvTransposeWithNorms(num_channels, 3, 2, 1, 1))
        for _ in range(num_filters_per_block):
            self.conv_layers.append(
                ConvTransposeWithNorms(num_channels, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x


class StepUpPyramid(nn.Module):

    def __init__(self, num_steps: int, num_channels: int, num_filters: int):
        super().__init__()
        assert num_steps > 0, f"num_steps must be > 0, got {num_steps}"
        assert num_channels > 0, f"num_channels must be > 0, got {num_channels}"
        assert num_filters > 0, f"num_filters must be > 0, got {num_filters}"

        self.step_up_layers = nn.ModuleList()
        for _ in range(num_steps):
            self.step_up_layers.append(StepUp(num_channels, num_filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step_up_layer in self.step_up_layers:
            x = step_up_layer(x)
        return x


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, pseudoimage_dims: tuple, input_num_channels: int,
                 num_filters_per_block: int, num_layers_of_pyramid: int):
        super().__init__()
        pseudoimage_x, pseudoimage_y, = pseudoimage_dims
        assert pseudoimage_x > 0, f"pseudoimage_x must be > 0, got {pseudoimage_x}"
        assert pseudoimage_y > 0, f"pseudoimage_y must be > 0, got {pseudoimage_y}"
        assert num_filters_per_block > 0, f"num_filters must be > 0, got {num_filters_per_block}"
        assert num_layers_of_pyramid > 0, f"num_layers must be > 0, got {num_layers_of_pyramid}"

        self.num_filters = num_filters_per_block
        self.num_layers = num_layers_of_pyramid

        self.down_layers = nn.ModuleList()
        # Create down layers
        for _ in range(num_layers_of_pyramid):
            self.down_layers.append(
                StepDown(input_num_channels, num_filters_per_block))

        # For each down layer, we need a separate up pyramid layer
        self.up_pyramids = nn.ModuleList()
        for idx in range(num_layers_of_pyramid):
            self.up_pyramids.append(
                StepUpPyramid(idx + 1, input_num_channels,
                              num_filters_per_block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_channels, height, width)
        # We need to downsample x num_layers times
        # Each downsampled layer will be used in the up pyramid
        downsampled_layers = []

        down_x = x
        for down_layer in self.down_layers:
            down_x = down_layer(down_x)
            downsampled_layers.append(down_x)

        # Now we need to upsample x num_layers times
        # Each upsampled layer will be concatenated with the corresponding
        # downsampled layer
        upsampled_layers = [x]
        for up_x, up_pyramid in zip(downsampled_layers, self.up_pyramids):
            upsampled_layers.append(up_pyramid(up_x))

        # Concatenate the upsampled layers together into a single tensor
        upsampled_latent = torch.cat(upsampled_layers, dim=1)
        return upsampled_latent
