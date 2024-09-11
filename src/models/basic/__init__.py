import torch
import torch.nn as nn

@torch.no_grad()
def cal_pose0to1(pose0: torch.Tensor, pose1: torch.Tensor):
    """
    Note(Qingwen 2023-12-05 11:09):
    Don't know why but it needed set the pose to float64 to calculate the inverse 
    otherwise it will be not expected result....
    """
    pose1_inv = torch.eye(4, dtype=torch.float64, device=pose1.device)
    pose1_inv[:3,:3] = pose1[:3,:3].T
    pose1_inv[:3,3] = (pose1[:3,:3].T * -pose1[:3,3]).sum(axis=1)
    pose_0to1 = pose1_inv @ pose0.type(torch.float64)
    return pose_0to1.type(torch.float32)

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