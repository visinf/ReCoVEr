import argparse
from pathlib import Path

from utils.parser import parse_args

from torch.distributed import all_reduce
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import calc_epe

from models.recover import ReCoVEr_MN, ReCoVEr_RN, ReCoVEr_CX

from data.dataloader import fetch_dataloader
from utils.utils import load_ckpt
from loss.sequence_loss import sequence_loss
from utils.ddp_utils import *
from tqdm import tqdm
import os

os.system("export KMP_INIT_AT_FORK=FALSE")


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""

    param_groups = [{"params": []}, {"params": []}]

    for name, param in model.named_parameters():
        if name.startswith("update_block.encoder.convc1"):
            param_groups[1]["params"].append(param)
        # else:
        elif not args.disable_cost:
            param_groups[0]["params"].append(param)
        elif "fnet" not in name:
            param_groups[0]["params"].append(param)
        else:
            print(name)

    optimizer = optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


@torch.no_grad
def validate(model, args, writer, step, rank, world_size, use_ddp):
    model.eval()

    val_loader = fetch_dataloader(
        args, rank=rank, world_size=world_size, use_ddp=use_ddp, train=False
    )

    epe_sum = torch.Tensor([0]).float().cuda(rank)
    epe_count = torch.Tensor([0]).int().cuda(rank)

    for i_batch, data_blob in (
        enumerate(tqdm(val_loader)) if rank == 0 else enumerate(val_loader)
    ):
        image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
        output = model(
            image1,
            image2,
            iters=args.iters,
            test_mode=True,
            disable_cost=args.disable_cost,
        )

        epe = calc_epe(output["final"], flow, valid)

        epe_sum += epe.sum()
        epe_count += epe.numel()

    all_reduce(epe_sum)
    all_reduce(epe_count)

    if rank == 0:
        writer.add_scalar("val/epe", epe_sum / epe_count, step)

    model.train()


def train(args, rank=0, world_size=1, use_ddp=False):
    """Full training loop"""
    device_id = rank

    if args.extractor["type"].lower() == "convnext":
        model = ReCoVEr_CX(args)
    elif args.extractor["type"].lower() == "mobilenetv3":
        model = ReCoVEr_MN(args)
    elif args.extractor["type"].lower() == "resnet":
        model = ReCoVEr_RN(args)

    model = model.to(device_id)
    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model
        )  # there might not be any, actually
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], static_graph=True
        )

    model.train()
    train_loader = fetch_dataloader(
        args, rank=rank, world_size=world_size, use_ddp=use_ddp
    )
    optimizer, scheduler = fetch_optimizer(args, model)

    if rank == 0:
        logger = SummaryWriter(str(Path(args.log_dir) / args.name))
    else:
        logger = None

    ckpt_dir = Path(args.log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0

    validate(model, args, logger, total_steps, rank, world_size, use_ddp)

    should_keep_training = True
    # torch.autograd.set_detect_anomaly(True)
    while should_keep_training:
        # shuffle sampler
        train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, data_blob in (
            enumerate(tqdm(train_loader)) if rank == 0 else enumerate(train_loader)
        ):
            optimizer.zero_grad()

            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model(
                image1,
                image2,
                flow_gt=flow,
                iters=args.iters,
                disable_cost=args.disable_cost,
            )
            loss = sequence_loss(output, flow, valid, args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = str(ckpt_dir / f"{total_steps + 1}_{args.name}.pth")
                torch.save(model.module.state_dict(), PATH)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                validate(model, args, logger, total_steps, rank, world_size, use_ddp)

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            if rank == 0:
                epe = calc_epe(output["final"], flow, valid).mean()

                logger.add_scalar("train/loss", loss, total_steps)
                logger.add_scalar("train/epe", epe, total_steps)
                logger.add_scalar("train/lr", scheduler.get_last_lr()[0], total_steps)
            total_steps += 1

    PATH = str(ckpt_dir / ("%s.pth" % args.name))
    if rank == 0:
        torch.save(model.module.state_dict(), PATH)
        logger.flush()

    return PATH


def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)


def get_next_exp_id(args):
    name = args.name
    log_dir = Path(args.log_dir)
    all_ids = [
        int(d.name.split("_exp")[-1])
        for d in log_dir.glob("*")
        if d.is_dir() and name in d.name
    ]
    if len(all_ids) == 0:
        return 1

    else:
        return max(all_ids) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    args = parse_args(parser)
    args.name += f"_exp{str(get_next_exp_id(args))}"
    smp, world_size = init_ddp()
    if world_size > 1:
        spwn_ctx = mp.spawn(
            main, nprocs=world_size, args=(world_size, args, True), join=False
        )
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")
