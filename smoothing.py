from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage.io import imsave


def lin_interp(n, p1, p2):
	# p1,p2 = 128,3
	x = np.zeros((n, 128, 3))

	for i in range(n):
		a = (i + 1) / (n + 1)
		x[i] = (1 - a) * p1 + a * p2

	return x


def smooth(in_img, ws, out_img):
	in_img = np.array(Image.open(in_img)) / 255.0
	orig_img = in_img[24:-24, :1280, :]
	in_img = in_img[:, 1280:, :]

	# 6,10,128,128,3
	patches = np.reshape(in_img, (6, 128, 10, 128, 3))
	patches = np.transpose(patches, (0, 2, 1, 3, 4))

	h = ws // 2

	for i in range(5):
		for j in range(10):
			p1 = patches[i, j][128 - h, :, :]
			p2 = patches[i + 1][j, h, :, :]

			x = lin_interp(ws, p1, p2)
			patches[i, j, 128 - h:, :, :] = x[:h, :, :]
			patches[i + 1, j, :h, :, :] = x[h:, :, :]

	for i in range(6):
		for j in range(9):
			p3 = patches[i, j][:, 128 - h, :]
			p4 = patches[i, j + 1][:, h, :]

			x = lin_interp(ws, p3, p4)
			patches[i, j][:, 128 - h:, :] = np.transpose(x[:h, :, :], (1, 0, 2))
			patches[i, j + 1][:, :h, :] = np.transpose(x[h:, :, :], (1, 0, 2))

	out = np.transpose(patches, (0, 2, 1, 3, 4))
	out = np.reshape(out, (768, 1280, 3))
	out = out[24:-24, :, :]

	out = np.concatenate((orig_img, out), axis=1)
	imsave(out_img, out)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_img', type=str, required=True)
	parser.add_argument('--out_img', type=str, required=True)
	parser.add_argument('--window_size', type=int, required=True)
	args = parser.parse_args()

	smooth(args.in_img, args.window_size, args.out_img)
