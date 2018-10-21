from torchvision.utils import save_image


def save_imgs(imgs, to_size, name):
	# x = np.array(x)
	# x = np.transpose(x, (1, 2, 0)) * 255
	# x = x.astype(np.uint8)
	# imsave(name, x)

	# x = 0.5 * (x + 1)

	# to_size = (C, H, W)
	imgs = imgs.clamp(0, 1)
	imgs = imgs.view(imgs.size(0), *to_size)
	save_image(imgs, name)
