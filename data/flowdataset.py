import numpy as np
import torch
import torch.utils.data as data

import random
import h5py

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from utils.static_transform import StaticTransform
from utils.utils import induced_flow, check_cycle_consistency


class FlowDataset(data.Dataset):
    def __init__(
        self,
        aug_params=None,
        sparse=False,
        return_gt_path=False,
        static_transforms=None,
    ):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = "unknown"
        self.subsample_groundtruth = False
        self.return_gt_path = return_gt_path
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

        if static_transforms is not None:
            self.static_transforms = StaticTransform(**static_transforms)
        else:
            self.static_transforms = None

    def __getitem__(self, index):
        # return self.fetch(index)

        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)

    def fetch(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            if self.dataset == "TartanAir":
                flow = np.load(self.flow_list[index])
                valid = np.load(self.mask_list[index])
                # rescale the valid mask to [0, 1]
                valid = 1 - valid / 100

            elif self.dataset == "MegaDepth":
                depth0 = np.array(h5py.File(self.extra_info[index][0], "r")["depth"])
                depth1 = np.array(h5py.File(self.extra_info[index][1], "r")["depth"])
                camera_data = self.megascene[index]
                flow_01, flow_10 = induced_flow(depth0, depth1, camera_data)
                valid = check_cycle_consistency(flow_01, flow_10)
                flow = flow_01
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            if self.dataset == "Infinigen":
                # Inifinigen flow is stored as a 3D numpy array, [Flow, Depth]
                flow = np.load(self.flow_list[index])
                flow = flow[..., :2]
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> flow will have same dimensions as images
            # used for spring dataset
            flow = flow[::2, ::2]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.static_transforms is not None:
            if self.sparse:
                img1, img2, flow, valid = self.static_transforms(
                    img1, img2, flow, valid
                )
            else:
                img1, img2, flow = self.static_transforms(img1, img2, flow)

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[torch.isnan(flow)] = 0
        flow[flow.abs() > 1e9] = 0

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.return_gt_path:
            return img1, img2, flow, valid.float(), self.flow_list[index]
        else:
            return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)
