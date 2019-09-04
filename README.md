# Lossy Image Compression with Compressive Autoencoders

See [wiki](https://github.com/alexandru-dinu/cae/wiki) for more details and further results.

## Results

`cae_32x32x32_zero_pad_bin` model, after roughly 5.8 millions of optimization steps;
left: original, right: reconstructed.

![](https://i.imgur.com/RM7xJ6W.png)
![](https://i.imgur.com/GWDbay4.png)
![](https://i.imgur.com/KNi7fkh.jpg)
![](https://i.imgur.com/LDSoBKb.jpg)
![](https://i.imgur.com/cBJbLKg.jpg)
![](https://i.imgur.com/ARbPB86.jpg)

## Resources

A dataset can be constructed by downloading frames using the scripts provided [here](https://github.com/gsssrao/youtube-8m-videos-frames).
For the above results, I have randomly selected and downloaded 121,827 frames.

## References
[1] [Lossy Image Compression with Compressive Autoencoders, Theis et al.](https://arxiv.org/abs/1703.00395)

[2] [Variable Rate Image Compression with Recurrent Neural Networks, Toderici et al.](http://arxiv.org/abs/1511.06085)
