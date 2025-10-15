import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models.recover import ReCoVEr_MN, ReCoVEr_RN, ReCoVEr_CX
from utils.visualize import flow_to_rgb


def read_img(file):
    img = np.array(Image.open(str(file)))
    img = torch.Tensor(img)
    img = img.permute(2, 0, 1)  # HWC -> CHW

    return img


def calc_flow(model, image1, image2, scale):
    img1 = F.interpolate(
        image1, scale_factor=2**scale, mode="bilinear", align_corners=False
    )
    img2 = F.interpolate(
        image2, scale_factor=2**scale, mode="bilinear", align_corners=False
    )
    H, W = image1.size(-2), image1.size(-1)

    flow = model(
        img1,
        img2,
        test_mode=True,
    )["final"]

    flow_down = F.interpolate(
        flow, size=(H, W), mode="bilinear", align_corners=False
    ) * (0.5**scale)

    return flow_down


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Demo", description="Calculates the optical flow of two input frames"
    )

    parser.add_argument("frame1", nargs="?", type=str)
    parser.add_argument("frame2", nargs="?", type=str)

    parser.add_argument(
        "--model",
        default="recover_cx",
        nargs="?",
        choices=["recover_cx", "recover_rn", "recover_mn"],
    )
    parser.add_argument("--scale", nargs="?", type=int, default=0)
    parser.add_argument("--out", nargs="?", type=str, required=False, default=None)
    parser.add_argument("--display", action="store_true")

    args = parser.parse_args()

    if args.model.lower() == "recover_cx":
        model = ReCoVEr_CX(pretrained=True)
    elif args.model.lower() == "recover_rn":
        model = ReCoVEr_RN(pretrained=True)
    elif args.model.lower() == "recover_mn":
        model = ReCoVEr_MN(pretrained=True)
    else:
        print(f"Unknown model: {args.model}")
        exit(1)

    model = model.cuda()
    model.eval()

    img1 = read_img(args.frame1).unsqueeze(0).cuda()
    img2 = read_img(args.frame2).unsqueeze(0).cuda()

    with torch.no_grad():
        flow = calc_flow(model, img1, img2, args.scale)
        vis = flow_to_rgb(flow)[0]
        np_vis = vis.permute(1, 2, 0).detach().cpu().numpy()

    if args.display:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(
            img1[0].permute(1, 2, 0).detach().cpu().numpy().round().astype(np.uint8)
        )
        plt.subplot(1, 3, 2)
        plt.imshow(
            img2[0].permute(1, 2, 0).detach().cpu().numpy().round().astype(np.uint8)
        )
        plt.subplot(1, 3, 3)
        plt.imshow(np_vis)
        plt.show()

    if args.out is not None:
        pil_img = Image.fromarray((255 * np_vis).round().astype(np.uint8))
        pil_img.save(args.out)
