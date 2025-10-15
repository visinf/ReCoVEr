import numpy as np
import cv2


class StaticTransform:
    def __init__(self, width=None):
        super().__init__()

        self.width = width

    def resize_sparse_flow_map(self, flow, valid, new_w, new_h):  # fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = new_h
        wd1 = new_w

        fx = wd1 / wd
        fy = ht1 / ht

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def __call__(self, img1, img2, flow, valid=None):
        return_valid = valid is not None

        if valid is None:
            valid = np.ones(img1.shape[:2])

        if self.width is not None:
            h, w = img1.shape[:2]
            new_w = self.width
            new_h = int(new_w / w * h)

            img1 = cv2.resize(
                img1, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            flow, valid = self.resize_sparse_flow_map(flow, valid, new_w, new_h)

        if return_valid:
            return img1, img2, flow, valid
        else:
            return img1, img2, flow
