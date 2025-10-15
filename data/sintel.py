from glob import glob
import os
import os.path as osp

from .flowdataset import FlowDataset


class MpiSintel(FlowDataset):
    def __init__(
        self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))
