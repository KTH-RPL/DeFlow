import torch
import torch.nn as nn

from typing import Tuple


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)


class BilinearDecoder(nn.Module):

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         scale_factor=self.scale_factor,
                                         mode="bilinear",
                                         align_corners=False)

class UpsampleSkip(nn.Module):

    def __init__(self, skip_channels: int, latent_channels: int,
                 out_channels: int):
        super().__init__()
        self.u1_u2 = nn.Sequential(
            nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
            BilinearDecoder(2))
        self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        self.u4_u5 = nn.Sequential(
            nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u2_res = self.u1_u2(a)
        u3_res = self.u3(b)
        u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        return u5_res


class FastFlow3DUNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(32, 64, 3, 2, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1))
        self.decoder_step1 = UpsampleSkip(512, 256, 256)
        self.decoder_step2 = UpsampleSkip(256, 128, 128)
        self.decoder_step3 = UpsampleSkip(128, 64, 64)
        self.decoder_step4 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, pc0_B: torch.Tensor,
                pc1_B: torch.Tensor) -> torch.Tensor:

        expected_channels = 32
        assert pc0_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc0_B.shape[1]}"
        assert pc1_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc1_B.shape[1]}"

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)

        Rstar = torch.cat([pc0_R, pc1_R],
                          dim=1)  # torch.Size([1, 512, 64, 64])
        Lstar = torch.cat([pc0_L, pc1_L],
                          dim=1)  # torch.Size([1, 256, 128, 128])
        Fstar = torch.cat([pc0_F, pc1_F],
                          dim=1)  # torch.Size([1, 128, 256, 256])
        Bstar = torch.cat([pc0_B, pc1_B],
                          dim=1)  # torch.Size([1, 64, 512, 512])

        S = self.decoder_step1(Rstar, Lstar)
        T = self.decoder_step2(S, Fstar)
        U = self.decoder_step3(T, Bstar)
        V = self.decoder_step4(U)

        return V