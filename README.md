# Compressive Autoencoder

[![Discussions](https://img.shields.io/badge/discussions-welcome-brightgreen)](https://github.com/alexandru-dinu/cae/discussions)
[![Wiki](https://img.shields.io/badge/docs-wiki-white)](https://github.com/alexandru-dinu/cae/wiki)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Getting started

The quickest way to start experimenting is to use [this model](https://drive.google.com/open?id=1SSek44svPAZClmOg8xX-DLDUxQdnK22A) trained on [this smaller dataset](https://drive.google.com/open?id=1wbwkpz38stSFMwgEKhoDCQCMiLLFVC4T). An arbitrary dataset can be constructed by downloading frames using the scripts provided [here](https://github.com/gsssrao/youtube-8m-videos-frames).

See [wiki](https://github.com/alexandru-dinu/cae/wiki) for more details, download links and further results.

### Training
```bash
python train.py --config ../configs/train.yaml
```

Example `train.yaml`:
```yaml
exp_name: training

num_epochs: 1
batch_size: 16
learning_rate: 0.0001

# start fresh
resume: false
checkpoint: null
start_epoch: 1

batch_every: 1
save_every: 10
epoch_every: 1
shuffle: true
dataset_path: datasets/yt_small_720p
num_workers: 2
device: cuda
```

### Testing
Given a trained model (`checkpoint`), perform inference on images @ `dataset_path`.

```bash
python test.py --config ../configs/test.yaml
```

Example `test.yaml`:
```yaml
exp_name: testing
checkpoint: model.state
batch_every: 100
shuffle: true
dataset_path: datasets/testing
num_workers: 1
device: cuda
```

**Note**: Currently, smoothing (i.e. linear interpolation in [`smoothing.py`](https://github.com/alexandru-dinu/cae/blob/master/src/smoothing.py#L19)) is used in order to account for the between-patches noisy areas due to padding (this still needs [further investigation](https://github.com/alexandru-dinu/cae/issues/20)).

## Results

- `cae_32x32x32_zero_pad_bin` model
- roughly 5.8 millions of optimization steps
- randomly selected and downloaded 121,827 frames
- left: original, right: reconstructed

![](https://i.imgur.com/RM7xJ6W.png)
![](https://i.imgur.com/GWDbay4.png)
![](https://i.imgur.com/KNi7fkh.jpg)
![](https://i.imgur.com/LDSoBKb.jpg)
![](https://i.imgur.com/cBJbLKg.jpg)
![](https://i.imgur.com/ARbPB86.jpg)

## References

- [1] [Lossy Image Compression with Compressive Autoencoders, Theis et al.](https://arxiv.org/abs/1703.00395)
- [2] [Variable Rate Image Compression with Recurrent Neural Networks, Toderici et al.](http://arxiv.org/abs/1511.06085)
