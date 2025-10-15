from glob import glob
import os.path as osp

from .flowdataset import FlowDataset


class TartanAir(FlowDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root="datasets/tartanair"):
        super(TartanAir, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = "TartanAir"
        self.root = root
        self._build_dataset()

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, "*/*/*"))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, "image_left/*.png")))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(
                    osp.join(scene, "flow", f"{frame0}_{frame1}_flow.npy")
                )
                self.mask_list.append(
                    osp.join(scene, "flow", f"{frame0}_{frame1}_mask.npy")
                )
