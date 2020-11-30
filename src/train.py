import os
import sys
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ImageFolder720p
from utils import save_imgs

from bagoftools.namespace import Namespace
from bagoftools.logger import Logger

# paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from models.cae_32x32x32_zero_pad_bin import CAE

logger = Logger(name=__name__, colorize=True)


def train(cfg: Namespace) -> None:
    assert cfg.device == 'cpu' or (cfg.device == 'cuda' and T.cuda.is_available())

    logger.info('training: experiment %s' % (cfg.exp_name))

    # make dir-tree
    exp_dir = ROOT_DIR / 'experiments' / cfg.exp_name

    for d in ['out', 'checkpoint', 'logs']:
        os.makedirs(exp_dir / d, exist_ok=True)

    cfg.to_file(exp_dir / 'train_config.txt')

    # tb writer
    writer = SummaryWriter(exp_dir / 'logs')

    model = CAE()
    model.train()
    if cfg.device == 'cuda':
        model.cuda()
    logger.info(f'loaded model on {cfg.device}')

    dataset = ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    logger.info('loaded dataset')

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_criterion = nn.MSELoss()

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # EPOCHS
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):
        # BATCHES
        for batch_idx, data in enumerate(dataloader, start=1):
            img, patches, _ = data

            if cfg.device == 'cuda':
                patches = patches.cuda()

            avg_loss_per_image = 0.0
            for i in range(6):
                for j in range(10):
                    optimizer.zero_grad()

                    x = patches[:, :, i, j, :, :]
                    y = model(x)
                    loss = loss_criterion(y, x)

                    avg_loss_per_image += (1 / 60) * loss.item()

                    loss.backward()
                    optimizer.step()

            avg_loss  += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.batch_every == 0:
                writer.add_scalar('train/avg_loss', avg_loss / cfg.batch_every, ts)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, ts)

                logger.debug(
                    '[%3d/%3d][%5d/%5d] avg_loss: %.8f' % (
                    epoch_idx, cfg.num_epochs, batch_idx, len(dataloader), avg_loss / cfg.batch_every))

                avg_loss = 0.0
                ts += 1
            # -- end batch every

            if batch_idx % cfg.save_every == 0:
                out = T.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = patches[0, :, i, j, :, :].unsqueeze(0).cuda()
                        out[i, j] = model(x).cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))

                y = T.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(imgs=y,
                    to_size=(3, 768, 2 * 1280),
                    name=exp_dir / f'out/{epoch_idx}_{batch_idx}.png')
            # -- end save every
        # -- end batches

        if epoch_idx % cfg.epoch_every == 0:
            epoch_avg /= (len(dataloader) * cfg.epoch_every)

            writer.add_scalar('train/epoch_avg_loss',
                avg_loss / cfg.batch_every, epoch_idx // cfg.epoch_every)

            logger.info('Epoch avg = %.8f' % epoch_avg)
            epoch_avg = 0.0

            T.save(model.state_dict(), exp_dir / f'checkpoint/model_{epoch_idx}.pth')
        # -- end epoch every
   # -- end epoch

    # save final model
    T.save(model.state_dict(), exp_dir / 'model_final.pth')

    # cleaning
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'rt') as fp:
        cfg = Namespace(**yaml.safe_load(fp))

    train(cfg)