import torch

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