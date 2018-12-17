import torch
import torch.nn as nn


# will see 3x128x128 patches
class AutoencoderConv(nn.Module):
	"""
	This AE module will be fed 3x128x128 patches from the original image

	Shapes are (batch_size, channels, height, width)
	"""

	def __init__(self):
		super(AutoencoderConv, self).__init__()

		self.enc_x = None

		# ENCODER

		# 64x64x64
		self.e_conv_1 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
			nn.LeakyReLU()
		)

		# 128x32x32
		self.e_conv_2 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
			nn.LeakyReLU()
		)

		# 128x32x32
		self.e_bb_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_bb_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_bb_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 16x16x16
		self.e_conv_3 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
			nn.Tanh()
		)

		# DECODER

		# 128x32x32
		self.d_subpix_1 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
		)

		# 128x32x32
		self.d_bb_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.d_bb_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.d_bb_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 256x64x64
		self.d_subpix_2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
		)

		# 3x128x128
		self.d_subpix_3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
			nn.Tanh()
		)

	def forward(self, x):
		# ENCODE
		ec1 = self.e_conv_1(x)
		ec2 = self.e_conv_2(ec1)
		ebb1 = self.e_bb_1(ec2) + ec2
		ebb2 = self.e_bb_2(ebb1) + ebb1
		ebb3 = self.e_bb_3(ebb2) + ebb2
		ec3 = self.e_conv_3(ebb3)  # in [-1, 1]

		with torch.no_grad():
			r = torch.rand(ec3.shape).cuda()
			p = (1 + ec3) / 2
			eps = torch.zeros(ec3.shape).cuda()
			eps[r <= p] = (1 - ec3)[r <= p]
			eps[r > p] = (-ec3 - 1)[r > p]

		# encoded tensor
		self.enc_x = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
		y = self.enc_x * 2.0 - 1  # (0|1) -> (-1|1)

		# DECODE
		cc1 = self.d_subpix_1(y)
		dbb1 = self.d_bb_1(cc1) + cc1
		dbb2 = self.d_bb_2(dbb1) + dbb1
		dbb3 = self.d_bb_3(dbb2) + dbb2
		cc2 = self.d_subpix_2(dbb3)
		dec = self.d_subpix_3(cc2)

		return dec
