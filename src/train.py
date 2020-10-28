import os
import sys
from argparse import Namespace

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader import ImageFolder720p
from utils import get_config, get_args, dump_cfg
from utils import save_imgs

from bagoftools.logger import Logger

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))
from cae_32x32x32_zero_pad_bin import CAE

logger = Logger(name='train', colorize=True)


def prologue(cfg: Namespace, *varargs) -> SummaryWriter:
    # sanity checks
    assert cfg.device == "cpu" or (cfg.device == "cuda" and T.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))

    # tb writer
    writer = SummaryWriter(f"{base_dir}/logs")

    return writer


def epilogue(cfg: Namespace, *varargs) -> None:
    writer = varargs[0]
    writer.close()


def train(cfg: Namespace) -> None:
    logger.info('starting training')

    # initial setup
    writer = prologue(cfg)

    # train-related code
    model = CAE()
    model.train()
    if cfg.device == "cuda":
        model.cuda()
    logger.debug(f"Model loaded on {cfg.device}")

    dataset = ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    logger.debug("Data loaded")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_criterion = nn.MSELoss()
    # scheduler = ...

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # train-loop
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):

        # scheduler.step()

        for batch_idx, data in enumerate(dataloader, start=1):
            img, patches, _ = data

            if cfg.device == "cuda":
                patches = patches.cuda()

            avg_loss_per_image = 0.0
            for i in range(6):
                for j in range(10):
                    optimizer.zero_grad()

                    x = Variable(patches[:, :, i, j, :, :])
                    y = model(x)
                    loss = loss_criterion(y, x)

                    avg_loss_per_image += (1 / 60) * loss.item()

                    loss.backward()
                    optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.batch_every == 0:
                writer.add_scalar("train/avg_loss", avg_loss / cfg.batch_every, ts)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, ts)

                logger.debug(
                    '[%3d/%3d][%5d/%5d] avg_loss: %.8f' %
                    (epoch_idx, cfg.num_epochs, batch_idx, len(dataloader), avg_loss / cfg.batch_every)
                )
                avg_loss = 0.0
                ts += 1

            if batch_idx % cfg.save_every == 0:
                out = T.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = Variable(patches[0, :, i, j, :, :].unsqueeze(0)).cuda()
                        out[i, j] = model(x).cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))

                y = T.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(imgs=y, to_size=(3, 768, 2 * 1280), name=f"../experiments/{cfg.exp_name}/out/out_{epoch_idx}_{batch_idx}.png")

        # -- batch-loop

        if epoch_idx % cfg.epoch_every == 0:
            epoch_avg /= (len(dataloader) * cfg.epoch_every)

            writer.add_scalar("train/epoch_avg_loss", avg_loss / cfg.batch_every, epoch_idx // cfg.epoch_every)

            logger.info("Epoch avg = %.8f" % epoch_avg)
            epoch_avg = 0.0
            T.save(model.state_dict(), f"../experiments/{cfg.exp_name}/chkpt/model_{epoch_idx}.pth")

    # -- train-loop

    # save final model
    T.save(model.state_dict(), f"../experiments/{cfg.exp_name}/model_final.pth")

    # final setup
    epilogue(cfg, writer)


if __name__ == '__main__':
    train(get_config(get_args()))