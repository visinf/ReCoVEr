import argparse
from collections import OrderedDict
import numpy as np
from pathlib import Path
from PIL import Image
import pprint
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.recover import ReCoVEr_MN, ReCoVEr_RN, ReCoVEr_CX

from metrics import EPE, PX1, PX3, PX5
from utils.flow_utils import flow_to_speed
from utils.loaders import flo, flo5, kitti, pfm


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


def read_img(file):
    img = np.array(Image.open(str(file)))
    img = torch.Tensor(img)

    if img.dim() == 3:
        img = img.permute(2, 0, 1)  # HWC -> CHW

    return img


def replace_state_dict_prefix(state_dict, prefix, new_prefix=""):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = new_prefix + k.removeprefix(prefix)

        new_state_dict[new_k] = v

    return new_state_dict


class SpringDataset(Dataset):
    def __init__(
        self,
        root="datasets/spring",
        split="train",
        side="left",
        direction="FW",
        subsample_gt=False,
    ):
        super().__init__()
        assert split in ["train", "val"]
        assert side in ["left", "right"]
        assert direction in ["FW", "BW"]

        self.subsample_gt = subsample_gt

        root = Path(root) / split

        self.data = []
        for scene in root.glob("*"):
            if not scene.is_dir():
                continue

            frames = sorted((scene / f"frame_{side}").glob("*.png"))

            scene_id = scene.name

            if direction == "FW":
                gen = range(len(frames) - 1)
            elif direction == "BW":
                gen = range(1, len(frames))

            for i in gen:
                frame_id = frames[i].name.split("_")[-1].split(".")[0]

                entry = {
                    "frame1": frames[i],
                    "frame2": frames[i + 1] if direction == "FW" else frames[i - 1],
                    "flow": (
                        root
                        / scene_id
                        / f"flow_{direction}_{side}"
                        / f"flow_{direction}_{side}_{frame_id}.flo5"
                    ),
                    "detailmap": (
                        root
                        / scene_id
                        / "maps"
                        / f"detailmap_flow_{direction}_{side}"
                        / f"detailmap_flow_{direction}_{side}_{frame_id}.png"
                    ),
                    "matchmap": (
                        root
                        / scene_id
                        / "maps"
                        / f"matchmap_flow_{direction}_{side}"
                        / f"matchmap_flow_{direction}_{side}_{frame_id}.png"
                    ),
                    "rigidmap": (
                        root
                        / scene_id
                        / "maps"
                        / f"rigidmap_{direction}_{side}"
                        / f"rigidmap_{direction}_{side}_{frame_id}.png"
                    ),
                    "skymap": (
                        root
                        / scene_id
                        / "maps"
                        / f"skymap_{side}"
                        / f"skymap_{side}_{frame_id}.png"
                    ),
                }

                self.data.append(entry)

    def __getitem__(self, item):
        data = self.data[item]

        frame1 = read_img(data["frame1"])
        frame2 = read_img(data["frame2"])
        flow = read_flow(data["flow"])

        detailmap = read_img(data["detailmap"]).bool()
        matchmap_rgb = read_img(data["matchmap"])
        matchmap = torch.ones((matchmap_rgb.size(-2), matchmap_rgb.size(-1))).bool()
        matchmap[(matchmap_rgb > 0).any(0)] = False

        rigidmap = read_img(data["rigidmap"]).bool()
        skymap = read_img(data["skymap"]).bool()

        if self.subsample_gt:
            flow = flow[:, ::2, ::2]
            rigidmap = rigidmap[::2, ::2]
            skymap = skymap[::2, ::2]
        else:
            detailmap = detailmap.repeat_interleave(2, -1).repeat_interleave(2, -2)
            matchmap = matchmap.repeat_interleave(2, -1).repeat_interleave(2, -2)

        valid = ~((torch.isnan(flow) | (flow.abs() > 1e9)).any(0))
        flow[:, ~valid] = 0

        return {
            "frame1": frame1,
            "frame2": frame2,
            "flow": flow,
            "valid": valid,
            "detailmap": detailmap,
            "matchmap": matchmap,
            "rigidmap": rigidmap,
            "skymap": skymap,
        }

    def __len__(self):
        return len(self.data)


def get_dataset():
    split = "train"
    direction = "ALL"
    side = "ALL"

    if split == "ALL":
        split = ["train", "val"]
    else:
        split = [split]

    if direction == "ALL":
        direction = ["FW", "BW"]
    else:
        direction = [direction]

    if side == "ALL":
        side = ["left", "right"]
    else:
        side = [side]

    dataset = None

    for sp in split:
        for dir in direction:
            for s in side:
                temp_dataset = SpringDataset(
                    split=sp, direction=dir, side=s, subsample_gt=True
                )
                if dataset is None:
                    dataset = temp_dataset
                else:
                    dataset = dataset + temp_dataset

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="recover_cx",
        nargs="?",
        choices=["recover_cx", "recover_rn", "recover_mn"],
    )
    parser.add_argument("--ckpt_file", default="", nargs="?", type=str)
    parser.add_argument("--batch_size", default=1, nargs="?", type=int)

    args = parser.parse_args()

    if args.model.lower() == "recover_cx":
        net = ReCoVEr_CX(pretrained=True)
    elif args.model.lower() == "recover_rn":
        net = ReCoVEr_RN(pretrained=True)
    elif args.model.lower() == "recover_mn":
        net = ReCoVEr_MN(pretrained=True)
    else:
        print(f"Unknown model: {args.model}")
        exit(1)

    if args.ckpt_file != "":
        state_dict = torch.load(args.ckpt_file, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]

        state_dict = replace_state_dict_prefix(state_dict, "model.")
        net.load_state_dict(state_dict, strict=True)

    dataset = get_dataset()

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    metrics = {
        k: [
            PX1(spring_mode=False),
            PX3(spring_mode=False),
            PX5(spring_mode=False),
            EPE(spring_mode=False),
        ]
        # k: [EPE(spring_mode=True)]
        for k in [
            "total",
            "low_detail",
            "high_detail",
            "matched",
            "unmatched",
            "rigid",
            "non_rigid",
            "s0-10",
            "s10-40",
            "s40+",
        ]
    }

    net = net.cuda()
    net.eval()

    for k in metrics.keys():
        for i in range(len(metrics[k])):
            metrics[k][i] = metrics[k][i].cuda()

    with torch.no_grad():
        for data in tqdm(dataloader):
            for key in data.keys():
                data[key] = data[key].cuda()

            speed = flow_to_speed(data["flow"])

            pred = net(data["frame1"], data["frame2"], test_mode=True)["final"]

            for m in metrics["total"]:
                m.update(pred, data["flow"], data["valid"])

            for m in metrics["low_detail"]:
                m.update(pred, data["flow"], data["valid"] & ~data["detailmap"])
            for m in metrics["high_detail"]:
                m.update(pred, data["flow"], data["valid"] & data["detailmap"])

            for m in metrics["matched"]:
                m.update(pred, data["flow"], data["valid"] & data["matchmap"])
            for m in metrics["unmatched"]:
                m.update(pred, data["flow"], data["valid"] & ~data["matchmap"])

            for m in metrics["rigid"]:
                m.update(pred, data["flow"], data["valid"] & ~data["rigidmap"])
            for m in metrics["non_rigid"]:
                m.update(pred, data["flow"], data["valid"] & data["rigidmap"])

            speed_mask = speed < 10
            for m in metrics["s0-10"]:
                m.update(pred, data["flow"], data["valid"] & speed_mask)

            speed_mask = (speed >= 10) & (speed < 40)
            for m in metrics["s10-40"]:
                m.update(pred, data["flow"], data["valid"] & speed_mask)

            speed_mask = speed >= 40
            for m in metrics["s40+"]:
                m.update(pred, data["flow"], data["valid"] & speed_mask)

    result = {}

    for k in metrics.keys():
        result[k] = {
            "PX1": metrics[k][0].compute().item(),
            "PX3": metrics[k][1].compute().item(),
            "PX5": metrics[k][2].compute().item(),
            "EPE": metrics[k][3].compute().item(),
        }

    pprint.pprint(result)
    # write_to_jsonl("spring_results_.jsonl", result)
