from glob import glob
import os
import random

from .flowdataset import FlowDataset


class SpringFlowDataset(FlowDataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """

    def __init__(
        self,
        aug_params=None,
        root="datasets/spring",
        split="train",
        subsample_groundtruth=True,
        static_transforms=None,
    ):
        super(SpringFlowDataset, self).__init__(
            aug_params, static_transforms=static_transforms
        )

        assert split in ["train", "val", "test", "val2"]

        if split != "val2":
            seq_root = os.path.join(root, split)
        else:
            seq_root = os.path.join(root, "train")

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == "test":
            self.is_test = True

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(
                    glob(os.path.join(seq_root, scene, f"frame_{cam}", "*.png"))
                )

                step = 1 if split != "val2" else 20

                # forward
                for frame in range(1, len(images), step):
                    self.data_list.append((frame, scene, cam, "FW"))

                # backward
                for frame in reversed(range(2, len(images) + 1, step)):
                    self.data_list.append((frame, scene, cam, "BW"))

        if split == "val2":
            random.Random(434).shuffle(self.data_list)

        for frame_data in self.data_list:
            frame, scene, cam, direction = frame_data

            img1_path = os.path.join(
                self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png"
            )

            if direction == "FW":
                img2_path = os.path.join(
                    self.seq_root,
                    scene,
                    f"frame_{cam}",
                    f"frame_{cam}_{frame+1:04d}.png",
                )
            else:
                img2_path = os.path.join(
                    self.seq_root,
                    scene,
                    f"frame_{cam}",
                    f"frame_{cam}_{frame-1:04d}.png",
                )

            self.image_list += [[img1_path, img2_path]]
            self.extra_info += [frame_data]

            if split != "test":
                flow_path = os.path.join(
                    self.seq_root,
                    scene,
                    f"flow_{direction}_{cam}",
                    f"flow_{direction}_{cam}_{frame:04d}.flo5",
                )
                self.flow_list += [flow_path]
