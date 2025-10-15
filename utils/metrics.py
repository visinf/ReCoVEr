import torch
import torch.nn.functional as F


def calc_epe(flow_pred, flow_gt, valid=None, max_flow=None, spring_mode=False):
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is not None:
        valid = valid >= 0.5

        if max_flow is not None:
            valid = valid & (mag < max_flow)

    elif max_flow is not None:
        valid = mag < max_flow
    else:
        valid = torch.ones_like(mag).bool()

    if spring_mode:
        flow_pred = flow_pred.repeat_interleave(2, -1).repeat_interleave(2, -2)

        if (
            valid is not None
            and valid.size(-1) == flow_pred.size(-1) // 2
            and valid.size(-2) == flow_pred.size(-2) // 2
        ):
            valid = valid.repeat_interleave(2, -1).repeat_interleave(2, -2)

    epe = torch.norm(flow_pred - flow_gt, p=2, dim=1)

    if spring_mode:
        epe = -F.max_pool2d(-epe, 2, 2)
        valid = F.max_pool2d(valid.float(), 2, 2).bool()

    return epe[valid]
