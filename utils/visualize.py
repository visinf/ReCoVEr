import numpy as np
import torch


class HSVWheel:
    def _hsv2rgb(self, h, s, v):
        input_shape = h.shape

        h = h.reshape(-1)
        s = s.reshape(-1)
        v = v.reshape(-1)

        i = np.int32(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        rgb = np.zeros((*input_shape, 3)).reshape(-1, 3)
        v, t, p, q = (
            v.reshape(-1, 1),
            t.reshape(-1, 1),
            p.reshape(-1, 1),
            q.reshape(-1, 1),
        )
        rgb[i == 0] = np.hstack([v, t, p])[i == 0]
        rgb[i == 1] = np.hstack([q, v, p])[i == 1]
        rgb[i == 2] = np.hstack([p, v, t])[i == 2]
        rgb[i == 3] = np.hstack([p, q, v])[i == 3]
        rgb[i == 4] = np.hstack([t, p, v])[i == 4]
        rgb[i == 5] = np.hstack([v, p, q])[i == 5]
        rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

        return rgb.reshape(*input_shape, 3)

    def hsv2rgb_torch(self, hsv: torch.Tensor) -> torch.Tensor:
        h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = v * s
        _x = _c * (-torch.abs(h * 6.0 % 2.0 - 1) + 1.0)
        _m = v - _c
        _o = torch.zeros_like(_c)
        idx = (h * 6.0).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb

    def get_color(self, angle, radius):
        angle = angle + 2 * np.pi
        angle = (angle % (2 * np.pi)) / (2 * np.pi)

        return self.hsv2rgb_torch(
            torch.stack([angle, radius, torch.ones_like(angle)], -3)
        )


def get_flow_max_radius(flow, per_row=False):
    complex_flow = torch.complex(flow[:, 1, ...], flow[:, 0, ...])
    radius = torch.abs(complex_flow)

    if per_row:
        max_radius = radius[~radius.isnan()].max(0)[0]
    else:
        max_radius = radius[~radius.isnan()].max()[0]

    return max_radius

    # return np.sqrt(np.nanmax(np.sum(flow**2, axis=2)))


def replace_nans(array, value=0):
    nan_mask = np.isnan(array)
    array[nan_mask] = value

    return array, nan_mask


hsv_wheel = HSVWheel()


def flow_to_rgb(
    flow: torch.Tensor, max_radius=None, normalize_per_row=True
) -> torch.Tensor:
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)

    flow = flow.float()

    complex_flow = torch.complex(flow[:, 1, ...], flow[:, 0, ...])
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)

    if max_radius is None:
        if normalize_per_row:
            max_radius = radius[~radius.isnan()].max(0)[0]
        else:
            max_radius = radius[~radius.isnan()].max()[0]

    radius /= max_radius + 1e-9

    return hsv_wheel.get_color(angle, radius)
