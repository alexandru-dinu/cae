import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder720p
from utils import dump_cfg, get_args, get_config, save_imgs

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))

from cae_32x32x32_zero_pad_bin import CAE


def prologue(cfg: Namespace, *varargs) -> None:
    # sanity checks
    assert cfg.chkpt not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)

    dump_cfg(f"{base_dir}/test_config.txt", vars(cfg))


def epilogue(cfg: Namespace, *varargs) -> None:
    pass


def test(cfg: Namespace) -> None:
    logger.info("=== Testing ===")

    # initial setup
    prologue(cfg)

    model = CAE()
    model.load_state_dict(torch.load(cfg.chkpt))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()

    logger.info("Loaded model")

    dataset = ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg.shuffle)

    logger.info("Loaded data")

    loss_criterion = nn.MSELoss()

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ = data
        if cfg.device == 'cuda':
            patches = patches.cuda()

        if batch_idx % cfg.batch_every == 0:
            pass

        out = torch.zeros(6, 10, 3, 128, 128)
        # enc = torch.zeros(6, 10, 16, 8, 8)
        avg_loss = 0

        for i in range(6):
            for j in range(10):
                x = Variable(patches[:, :, i, j, :, :]).cuda()
                y = model(x)
                out[i, j] = y.data

                loss = loss_criterion(y, x)
                avg_loss += (1 / 60) * loss.item()

        logger.debug('[%5d/%5d] avg_loss: %f' % (batch_idx, len(dataloader), avg_loss))

        # save output
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (768, 1280, 3))
        out = np.transpose(out, (2, 0, 1))

        y = torch.cat((img[0], out), dim=2)
        save_imgs(imgs=y.unsqueeze(0), to_size=(3, 768, 2 * 1280), name=f"../experiments/{cfg.exp_name}/out/test_{batch_idx}.png")

    # final setup
    epilogue(cfg)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)

    test(config)
