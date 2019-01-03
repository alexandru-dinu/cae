import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from image_folder import ImageFolder720p
from models.cae_32x32x32_zero_pad_bin import CAE
from utils import save_imgs


def train(cfg):
	os.makedirs(f"out/{cfg['exp_name']}", exist_ok=True)
	os.makedirs(f"checkpoints/{cfg['exp_name']}", exist_ok=True)

	# dump config for current experiment
	with open(f"checkpoints/{cfg['exp_name']}/setup.cfg", "wt") as f:
		for k, v in cfg.items():
			f.write("%15s: %s\n" % (k, v))

	model = CAE().cuda()

	if cfg['load']:
		model.load_state_dict(torch.load(cfg['chkpt']))
		logger.info("Loaded model from", cfg['chkpt'])

	model.train()
	logger.info("Done setup model")

	dataset = ImageFolder720p(cfg['dataset_path'])
	dataloader = DataLoader(
		dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers']
	)
	logger.info(f"Done setup dataloader: {len(dataloader)} batches of size {cfg['batch_size']}")

	mse_loss = nn.MSELoss()
	adam = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
	sgd = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'])

	optimizer = adam

	ra = 0

	for ei in range(cfg['resume_epoch'], cfg['num_epochs']):
		for bi, (img, patches, _) in enumerate(dataloader):

			avg_loss = 0
			for i in range(6):
				for j in range(10):
					x = Variable(patches[:, :, i, j, :, :]).cuda()
					y = model(x)
					loss = mse_loss(y, x)

					avg_loss += (1 / 60) * loss.item()

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

			ra = avg_loss if bi == 0 else ra * bi / (bi + 1) + avg_loss / (bi + 1)

			logger.debug(
				'[%3d/%3d][%5d/%5d] avg_loss: %f, ra: %f' %
				(ei + 1, cfg['num_epochs'], bi + 1, len(dataloader), avg_loss, ra)
			)

			# save img
			if bi % cfg['out_every'] == 0:
				out = torch.zeros(6, 10, 3, 128, 128)
				for i in range(6):
					for j in range(10):
						x = Variable(patches[0, :, i, j, :, :].unsqueeze(0)).cuda()
						out[i, j] = model(x).cpu().data

				out = np.transpose(out, (0, 3, 1, 4, 2))
				out = np.reshape(out, (768, 1280, 3))
				out = np.transpose(out, (2, 0, 1))

				y = torch.cat((img[0], out), dim=2).unsqueeze(0)
				save_imgs(imgs=y, to_size=(3, 768, 2 * 1280), name=f"out/{cfg['exp_name']}/out_{ei}_{bi}.png")

			# save model
			if bi % cfg['save_every'] == cfg['save_every'] - 1:
				torch.save(model.state_dict(), f"checkpoints/{cfg['exp_name']}/model_{ei}_{bi}.state")

	# save final model
	torch.save(model.state_dict(), f"checkpoints/{cfg['exp_name']}/model_final.state")


def main(args):
	train(cfg=json.load(open(args.cfg, "rt")))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, required=True)
	main(parser.parse_args())
