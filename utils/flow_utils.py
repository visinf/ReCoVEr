import torch

from pathlib import Path

from utils.loaders import *


def read_flow(file):
    file = Path(file)
    ext = file.name.split(".")[-1].lower()

    if ext == "flo":
        return flo(file)
    elif ext == "flo5":
        return flo5(file)
    elif ext == "png":
        return kitti(file)
    elif ext == "pfm":
        return pfm(file)

    raise NotImplemented


def flow_to_speed(flow):
    if flow.dim() == 3:
        return torch.norm(flow, 2, dim=0)
    elif flow.dim() == 4:
        return torch.norm(flow, 2, dim=1)

    raise NotImplemented
