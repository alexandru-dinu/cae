import argparse
import os
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from cae import CAE
from data_loader import ImageFolder720p
from logger import Logger
from namespace import Namespace
from utils import save_imgs

ROOT_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"

logger = Logger(__name__, colorize=True)


def test(cfg: Namespace) -> None:
    assert cfg.checkpoint not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and T.cuda.is_available())

    exp_dir = ROOT_EXP_DIR / cfg.exp_name
    os.makedirs(exp_dir / "out", exist_ok=True)
    cfg.to_file(exp_dir / "test_config.json")
    logger.info(f"[exp dir={exp_dir}]")

    model = CAE()
    model.load_state_dict(T.load(cfg.checkpoint))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()
    logger.info(f"[model={cfg.checkpoint}] on {cfg.device}")

    dataloader = DataLoader(
        dataset=ImageFolder720p(cfg.dataset_path),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    logger.info(f"[dataset={cfg.dataset_path}]")

    loss_criterion = nn.MSELoss()

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ = data
        if cfg.device == "cuda":
            patches = patches.cuda()

        """
        img:    6 x 10 x 3 x 128 x 128
        latent: 6 x 10 x 32 x 32 x 32
        """
        reconstructed_img = T.zeros(6, 10, 3, 128, 128)
        encoded_data = T.zeros(6, 10, *model.encoded_shape)
        avg_loss = 0

        for i in range(6):
            for j in range(10):
                x = patches[:, :, i, j, :, :].cuda()

                y_enc = model.encode(x)
                y_dec = model.decode(y_enc)

                encoded_data[i, j] = y_enc.data
                reconstructed_img[i, j] = y_dec.data

                loss = loss_criterion(y_dec, x)
                avg_loss += (1 / 60) * loss.item()

        logger.debug("[%5d/%5d] avg_loss: %f", batch_idx, len(dataloader), avg_loss)

        # save output
        reconstructed_img = np.transpose(reconstructed_img, (0, 3, 1, 4, 2))
        reconstructed_img = np.reshape(reconstructed_img, (768, 1280, 3))
        reconstructed_img = np.transpose(reconstructed_img, (2, 0, 1))

        # TODO: make custom file-type (header, packing etc.)
        T.save(encoded_data, exp_dir / f"out/enc_{batch_idx}.pt")

        both = T.cat((img[0], reconstructed_img), dim=2)
        save_imgs(
            imgs=both.unsqueeze(0),
            to_size=(3, 768, 2 * 1280),
            path=exp_dir / f"out/test_{batch_idx}.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "rt") as fp:
        cfg = Namespace(**yaml.safe_load(fp))

    test(cfg)
