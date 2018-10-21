# lossy image compression with compressive autoencoders

## arch
These models are inspired from [1].

As input, we have raw 720p images from [YouTube-8M dataset](https://research.google.com/youtube8m/) (credit goes to [gsssrao](https://github.com/gsssrao/youtube-8m-videos-frames) for the downloader and frames generator scripts). The dataset consists of 121,827 frames.
The images are padded to 1280x768 (i.e. 24,24 height pad), so that they can be split into 60 128x128 patches.
The model only gets to see a singular patch at a time; the loss is computed as `MSELoss(orig_ij, out_ij)` (thus, there are 60 optimization steps per image).

Before I get the chance to better document the code, here is a short description of each model:

 - `conv_32x32x32_bin`  - latent size is `32x32x32` bits/patch (i.e. compressed size: 240KB)
 - `conv_bin` - latent size is `16x8x8` bits/patch (i.e. compressed size: 7.5KB)
 - `conv_refl_pad_bin` - same as above, only that reflection pad is used (as opposed to zero pad)
 - `conv_512_bin` - latent size is `16x16x16` bits/patch (i.e. compressed size: 30KB)

[1] https://arxiv.org/abs/1703.00395

The documentation and further work will be written in the repo's [wiki](https://github.com/alexandru-dinu/cae/wiki).

## results

![1](https://i.imgur.com/GWDbay4.png)
![2](https://i.imgur.com/KNi7fkh.jpg)
![3](https://i.imgur.com/LDSoBKb.jpg)
![4](https://i.imgur.com/cBJbLKg.jpg)
![5](https://i.imgur.com/ARbPB86.jpg)
