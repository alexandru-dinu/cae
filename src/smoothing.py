import argparse
import os

import numpy as np
from PIL import Image
from skimage.io import imsave


def lin_interp(n, p1, p2):
    x = np.zeros((n, p1.shape[0], 128, 3))

    for i in range(n):
        a = (i + 1) / (n + 1)
        x[i] = (1 - a) * p1 + a * p2

    return x


def smooth(in_img, ws):
    _name, _ext = os.path.splitext(in_img)
    out_img = f"{_name}_s{ws}{_ext}"

    in_img = np.array(Image.open(in_img)) / 255.0
    orig_img = in_img[24:-24, :1280, :]  # left image, remove borders
    in_img = in_img[:, 1280:, :]  # right image

    # 6,10,128,128,3
    patches = np.reshape(in_img, (6, 128, 10, 128, 3))
    patches = np.transpose(patches, (0, 2, 1, 3, 4))

    h = ws // 2

    for i in range(5):
        p1 = patches[i, :, 128 - h, :, :]
        p2 = patches[i + 1, :, h, :, :]

        x = lin_interp(ws, p1, p2)
        patches[i, :, 128 - h :, :, :] = np.transpose(x[:h, :, :, :], (1, 0, 2, 3))
        patches[i + 1, :, :h, :, :] = np.transpose(x[h:, :, :, :], (1, 0, 2, 3))

    for j in range(9):
        p3 = patches[:, j, :, 128 - h, :]
        p4 = patches[:, j + 1, :, h, :]

        x = lin_interp(ws, p3, p4)
        patches[:, j, :, 128 - h :, :] = np.transpose(x[:h, :, :, :], (1, 2, 0, 3))
        patches[:, j + 1, :, :h, :] = np.transpose(x[h:, :, :, :], (1, 2, 0, 3))

    out = np.transpose(patches, (0, 2, 1, 3, 4))
    out = np.reshape(out, (768, 1280, 3))
    out = out[24:-24, :, :]

    out = np.concatenate((orig_img, out), axis=1)
    imsave(out_img, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_img", type=str, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    args = parser.parse_args()

    # make sure an even size is used
    args.window_size += args.window_size % 2

    smooth(args.in_img, args.window_size)
