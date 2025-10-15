import torch
from torchmetrics import Metric

from .functional import calc_epe


class PX(Metric):
    def __init__(self, threshold, max_flow=None, spring_mode=False):
        super().__init__()
        self.add_state("hits", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).long(), dist_reduce_fx="sum")

        self.threshold = threshold
        self.max_flow = max_flow
        self.spring_mode = spring_mode

    def update(self, flow_pred, flow_gt, valid=None):
        epe = calc_epe(flow_pred, flow_gt, valid, self.max_flow, self.spring_mode)

        hits = epe > self.threshold

        self.total += epe.numel()
        self.hits += hits.sum()

    def compute(self):
        return 100 * self.hits / self.total


class PX1(PX):
    def __init__(self, max_flow=None, spring_mode=False):
        super().__init__(1, max_flow, spring_mode)


class PX3(PX):
    def __init__(self, max_flow=None, spring_mode=False):
        super().__init__(3, max_flow, spring_mode)


class PX5(PX):
    def __init__(self, max_flow=None, spring_mode=False):
        super().__init__(5, max_flow, spring_mode)
