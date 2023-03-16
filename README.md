# DIRE for Diffusion-Generated Image Detection
<b>Zhendong Wang, <a href='https://jianminbao.github.io/'>Jianmin Bao</a>, <a href='http://staff.ustc.edu.cn/~zhwg/'>Wengang Zhou</a>, Weilun Wang, Hezhen Hu, Hong Chen, <a href='http://staff.ustc.edu.cn/~lihq/en/'>Houqiang Li </a> </b>

[[Paper (Comming Soon)]()] [[Dataset (Comming Soon)]()]


## Abstract
> Diffusion models have shown remarkable success in visual synthesis, but have also raised concerns about potential abuse for malicious purposes. In this paper, we seek to build a detector for telling apart real images from diffusion-generated images. We find that existing detectors struggle to detect images generated by diffusion models, even if we include generated images from a specific diffusion model in their training data. To address this issue, we propose a novel image representation called DIffusion Reconstruction Error (DIRE), which measures the error between an input image and its reconstruction counterpart by a pre-trained diffusion model. We observe that diffusion-generated images can be approximately reconstructed by a diffusion model while real images cannot. It provides a hint that DIRE can serve as a bridge to distinguish generated and real images. DIRE provides an effective way to detect images generated by most diffusion models, and it is general for detecting generated images from unseen diffusion models and robust to various perturbations. Furthermore, we establish a comprehensive diffusion-generated benchmark including images generated by eight diffusion models to evaluate the performance of diffusion-generated image detectors. Extensive experiments on our collected benchmark demonstrate that DIRE exhibits superiority over previous generated-image detectors.

<p align="center">
<img src="figs/teaser.png" >
</p>

## DIRE
<p align="center">
<img src="figs/dire.png" >
</p>

## TODO
- [ ] Release code.
- [ ] Release dataset.

## Acknowledgments
Our code is developed based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [CNNDetection](https://github.com/peterwang512/CNNDetection). Thanks for their sharing codes and models.