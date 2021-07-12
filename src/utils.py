import struct
from os import PathLike
from typing import Tuple, Union

import numpy as np
import torch as T
import torchvision


def save_imgs(imgs: T.Tensor, to_size: Tuple[int], path: Union[str, PathLike]) -> None:
    # x = np.array(x)
    # x = np.transpose(x, (1, 2, 0)) * 255
    # x = x.astype(np.uint8)
    # imsave(name, x)

    # x = 0.5 * (x + 1)

    # to_size = (C, H, W)
    imgs = imgs.clamp(0, 1).view(imgs.size(0), *to_size)
    torchvision.utils.save_image(imgs, path)


def save_encoded(enc: np.ndarray, fname: str) -> None:
    enc = np.reshape(enc, -1)
    sz = str(len(enc)) + "d"

    with open(fname, "wb") as fp:
        fp.write(struct.pack(sz, *enc))
