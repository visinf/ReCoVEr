import numpy as np
import struct
import torch

TAG_FLOAT = 202021.25  # tag to check the sanity of the file
TAG_STRING = "PIEH"  # string containing the tag
MIN_WIDTH = 1
MAX_WIDTH = 99999
MIN_HEIGHT = 1
MAX_HEIGHT = 99999


def flo(file):
    # open the file and read it
    with open(file, "rb") as f:
        # check tag
        tag = struct.unpack("f", f.read(4))[0]
        if tag != TAG_FLOAT:
            raise Exception("flow_utils.readFlowFile({:s}): wrong tag".format(file))
        # read dimension
        w, h = struct.unpack("ii", f.read(8))
        if w < MIN_WIDTH or w > MAX_WIDTH:
            raise Exception(
                "flow_utils.readFlowFile({:s}: illegal width {:d}".format(file, w)
            )
        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            raise Exception(
                "flow_utils.readFlowFile({:s}: illegal height {:d}".format(file, h)
            )
        flow = np.fromfile(f, "float32")
        if not flow.shape == (h * w * 2,):
            raise Exception(
                "flow_utils.readFlowFile({:s}: illegal size of the file".format(file)
            )
    flow.shape = (h, w, 2)

    return torch.Tensor(flow).permute(2, 0, 1)
