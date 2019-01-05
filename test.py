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


def test(cfg):
	os.makedirs(f"./test/{cfg['exp_name']}", exist_ok=True)

	model = CAE().cuda()

	model.load_state_dict(torch.load(cfg['chkpt']))
	model.eval()
	logger.info("Loaded model from", cfg['chkpt'])

	dataset = ImageFolder720p(cfg['dataset_path'])
	dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg['shuffle'])
	logger.info(f"Done setup dataloader: {len(dataloader)}")

	mse_loss = nn.MSELoss()

	for bi, (img, patches, path) in enumerate(dataloader):

		out = torch.zeros(6, 10, 3, 128, 128)
		# enc = torch.zeros(6, 10, 16, 8, 8)
		avg_loss = 0

		for i in range(6):
			for j in range(10):
				x = Variable(patches[:, :, i, j, :, :]).cuda()
				y = model(x)

				# e = model.enc_x.data
				# p = torch.tensor(np.random.permutation(e.reshape(-1, 1)).reshape(1, 16, 8, 8)).cuda()
				# out[i, j] = model.decode(p).data

				# enc[i, j] = model.enc_x.data
				out[i, j] = y.data

				loss = mse_loss(y, x)
				avg_loss += (1 / 60) * loss.item()

		logger.debug('[%5d/%5d] avg_loss: %f' % (bi, len(dataloader), avg_loss))

		# save output
		out = np.transpose(out, (0, 3, 1, 4, 2))
		out = np.reshape(out, (768, 1280, 3))
		out = np.transpose(out, (2, 0, 1))

		y = torch.cat((img[0], out), dim=2)
		save_imgs(imgs=y.unsqueeze(0), to_size=(3, 768, 2 * 1280), name=f"./test/{cfg['exp_name']}/test_{bi}.png")


# save encoded
# enc = np.reshape(enc, -1)
# sz = str(len(enc)) + 'd'
# open(f"./{cfg['exp_name']}/test_{bi}.enc", "wb").write(struct.pack(sz, *enc))

def main(args):
	cfg = json.load(open(args.cfg, "rt"))
	test(cfg)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, required=True)
	main(parser.parse_args())
