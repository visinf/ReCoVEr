import h5py
import torch


def flo5(file):
    with h5py.File(str(file), "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {file} does not have a 'flow' key.")
        return torch.Tensor(f["flow"][()]).permute(2, 0, 1)
