# Lossy Image Compression with Compressive Autoencoders

The quickest way to start experimenting is to use [this model](https://drive.google.com/open?id=1SSek44svPAZClmOg8xX-DLDUxQdnK22A) trained on [this smaller dataset](https://drive.google.com/open?id=1wbwkpz38stSFMwgEKhoDCQCMiLLFVC4T). An arbitrary dataset can be constructed by downloading frames using the scripts provided [here](https://github.com/gsssrao/youtube-8m-videos-frames).

See [wiki](https://github.com/alexandru-dinu/cae/wiki) for more details, download links and further results.

## Results

* `cae_32x32x32_zero_pad_bin` model
* roughly 5.8 millions of optimization steps
* randomly selected and downloaded 121,827 frames
* left: original, right: reconstructed

![](https://i.imgur.com/RM7xJ6W.png)
![](https://i.imgur.com/GWDbay4.png)
![](https://i.imgur.com/KNi7fkh.jpg)
![](https://i.imgur.com/LDSoBKb.jpg)
![](https://i.imgur.com/cBJbLKg.jpg)
![](https://i.imgur.com/ARbPB86.jpg)

## References

- [1] [Lossy Image Compression with Compressive Autoencoders, Theis et al.](https://arxiv.org/abs/1703.00395)
- [2] [Variable Rate Image Compression with Recurrent Neural Networks, Toderici et al.](http://arxiv.org/abs/1511.06085)
