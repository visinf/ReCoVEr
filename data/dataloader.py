import torch
from torch.utils.data import DataLoader

from .chairs import FlyingChairs
from .hd1k import HD1K
from .kitti import KITTI
from .sintel import MpiSintel
from .spring import SpringFlowDataset
from .tartan import TartanAir
from .things import FlyingThings3D

from utils.ddp_utils import init_fn, calc_num_workers


def fetch_dataloader(args, rank=0, world_size=1, use_ddp=False, train=True):
    """Create the data loader for the corresponding trainign set"""
    if train:
        if args.dataset == "chairs":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.1,
                "max_scale": args.scale + 1.0,
                "do_flip": True,
            }
            train_dataset = FlyingChairs(aug_params, split="training")

        elif args.dataset == "things":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.4,
                "max_scale": args.scale + 0.8,
                "do_flip": True,
            }
            clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
            final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
            train_dataset = clean_dataset + final_dataset

        elif args.dataset == "sintel":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.2,
                "max_scale": args.scale + 0.6,
                "do_flip": True,
            }
            sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
            sintel_final = MpiSintel(aug_params, split="training", dstype="final")
            train_dataset = sintel_clean + sintel_final

        elif args.dataset == "kitti":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.2,
                "max_scale": args.scale + 0.4,
                "do_flip": False,
            }
            train_dataset = KITTI(aug_params, split="training")

        elif args.dataset == "spring":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale,
                "max_scale": args.scale + 0.2,
                "do_flip": True,
            }
            train_dataset = SpringFlowDataset(aug_params, subsample_groundtruth=True)

        elif args.dataset == "TartanAir":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.2,
                "max_scale": args.scale + 0.4,
                "do_flip": True,
            }
            train_dataset = TartanAir(aug_params)

        elif args.dataset == "TSKH":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.2,
                "max_scale": args.scale + 0.6,
                "do_flip": True,
            }
            things = FlyingThings3D(aug_params, dstype="frames_cleanpass")
            sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
            sintel_final = MpiSintel(aug_params, split="training", dstype="final")
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": args.scale - 0.3,
                    "max_scale": args.scale + 0.5,
                    "do_flip": True,
                }
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": args.scale - 0.5,
                    "max_scale": args.scale + 0.2,
                    "do_flip": True,
                }
            )
            train_dataset = (
                20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things
            )

        elif args.dataset == "TKH":
            aug_params = {
                "crop_size": args.image_size,
                "min_scale": args.scale - 0.4,
                "max_scale": args.scale + 0.8,
                "do_flip": True,
            }
            clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
            final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": args.scale - 0.3,
                    "max_scale": args.scale + 0.5,
                    "do_flip": True,
                }
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": args.scale - 0.5,
                    "max_scale": args.scale + 0.2,
                    "do_flip": True,
                }
            )
            train_dataset = 100 * hd1k + clean_dataset + final_dataset + 1000 * kitti

        if use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )
            num_gpu = torch.cuda.device_count()
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size // num_gpu,
                shuffle=(train_sampler is None),
                num_workers=calc_num_workers(),
                sampler=train_sampler,
                worker_init_fn=init_fn,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                pin_memory=False,
                shuffle=True,
                # num_workers=32,
                num_workers=8,
                drop_last=True,
            )

        print("Training with %d image pairs" % len(train_dataset))
        return train_loader

    if args.val_dataset == "spring":
        val_dataset = SpringFlowDataset(split="val2")

    print("Validating with %d image pairs" % len(val_dataset))

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank
        )
        num_gpu = torch.cuda.device_count()
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size // num_gpu,
            shuffle=(train_sampler is None),
            num_workers=calc_num_workers(),
            sampler=train_sampler,
            worker_init_fn=init_fn,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            pin_memory=False,
            shuffle=False,
            # num_workers=32,
            num_workers=8,
            drop_last=True,
        )

    return val_loader
