import cv2
import numpy as np
import torch


def kitti(file):
    flow = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0

    return torch.Tensor(flow).permute(2, 0, 1), torch.Tensor(valid).bool()
