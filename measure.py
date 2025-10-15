import argparse
import numpy as np
import torch

from models.recover import ReCoVEr_MN, ReCoVEr_RN, ReCoVEr_CX


def measure_peak_memory(model, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    _ = model(*args, **kwargs)

    memory = torch.cuda.max_memory_allocated()

    return {
        "B": memory,
        "KB": memory / 1024,
        "MB": memory / 1024 / 1024,
        "GB": memory / 1024 / 1024 / 1024,
        "TB": memory / 1024 / 1024 / 1024 / 1024,
    }


def measure_time(model, warmup_it, measure_it, *args, **kwargs):
    times = []
    for _ in range(measure_it + warmup_it):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(*args, **kwargs)

        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    times = times[warmup_it:]

    mean = np.mean(times).item()
    std = np.std(times).item()

    return {"mean": mean, "std": std, "measurements": times}


def measure_flops(model, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
    ) as prof:
        _ = model(*args, **kwargs)

    events = prof.events()
    flops = sum([int(evt.flops) for evt in events])

    return {
        "FLOPS": flops,
        "KFLOPS": flops / 1000,
        "MFLOPS": flops / 1000 / 1000,
        "GFLOPS": flops / 1000 / 1000 / 1000,
        "TFLOPS": flops / 1000 / 1000 / 1000 / 1000,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Measure", description="Measures the required FLOPS, runtime, and memory"
    )

    parser.add_argument("width", nargs="?", type=int)
    parser.add_argument("height", nargs="?", type=int)
    parser.add_argument("--warmup_it", nargs="?", type=int, default=10)
    parser.add_argument("--measure_it", nargs="?", type=int, default=100)
    parser.add_argument(
        "--model",
        default="recover_cx",
        nargs="?",
        choices=["recover_cx", "recover_rn", "recover_mn"],
    )

    args = parser.parse_args()

    if args.model.lower() == "recover_cx":
        model = ReCoVEr_CX()
    elif args.model.lower() == "recover_rn":
        model = ReCoVEr_RN()
    elif args.model.lower() == "recover_mn":
        model = ReCoVEr_MN()
    else:
        print(f"Unknown model: {args.model}")
        exit(1)

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        input1 = torch.rand((1, 3, args.height, args.width)).cuda()
        input2 = torch.rand((1, 3, args.height, args.width)).cuda()

        memory = measure_peak_memory(model, input1, input2, test_mode=True)
        time = measure_time(
            model,
            args.warmup_it,
            args.measure_it,
            input1,
            input2,
            test_mode=True,
        )
        flops = measure_flops(model, input1, input2, test_mode=True)

        print(
            f'{args.model} h:{args.height} w:{args.width} memory:{memory["GB"]:.2f}GB time:{time["mean"]:.2f}ms FLOPS:{flops["GFLOPS"]:.2f}GFLOPS'
        )
