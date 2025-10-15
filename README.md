<div align="center">
<h1>Removing Cost Volumes from Optical Flow Estimators</h1>

[**Simon Kiefhaber**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/visinf_team_details_125120.en.jsp)<sup>1,2</sup> &nbsp;&nbsp;&nbsp;
[**Stefan Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>1,2</sup> &nbsp;&nbsp;&nbsp;
[**Simone Schaub-Meyer**](https://schaubsi.github.io/)<sup>1,2</sup>

<sup>1</sup>TU Darmstadt &nbsp;&nbsp;&nbsp;
<sup>2</sup>hessian.AI &nbsp;&nbsp;&nbsp;

<h3>ICCV 2025 Oral</h3>

<a href="https://visinf.github.io/recover/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<center>
    <img src="https://visinf.github.io/recover/static/images/teaser_ours.webp" width="100%">
</center>
</div>

## Abstract
Cost volumes are used in every modern optical flow estimator, but due to their computational and space complexity, they are often a limiting factor regarding both processing speed and the resolution of input frames. Motivated by our empirical observation that cost volumes lose their importance once all other network parts of, e.g., a RAFT-based pipeline have been sufficiently trained, we introduce a training strategy that allows removing the cost volume from optical flow estimators throughout training. This leads to significantly improved inference speed and reduced memory requirements. Using our training strategy, we create three different models covering different compute budgets. Our most accurate model reaches state-of-the-art accuracy while being $1.2\times$ faster and having a $6\times$ lower memory footprint than comparable models; our fastest model is capable of processing Full HD frames at $20\,\mathrm{FPS}$ using only $500\,\mathrm{MB}$ of GPU memory.

## Setup
### Install Requirements
```
conda create -n recover python=3.10
conda activate recover
pip install -r requirements.txt
```

### Datasets
For evaluation and training the datasets have to be located in `datasets/`. The following structure is expected:
```
├── datasets
    ├── Sintel
    ├── KITTI
    ├── FlyingChairs
    ├── FlyingThings3D
    ├── HD1K
    ├── spring
    ├── tartanair
```

### Checkpoints
Pre-trained checkpoints can be downloaded [here](https://github.com/visinf/ReCoVEr/releases/download/0.1/ckpt.tar.gz) or using the following command:
```
curl -s -L https://github.com/visinf/ReCoVEr/releases/download/0.1/ckpt.tar.gz | tar xz
```

## Demo
After downloading and unpacking our checkpoints to `ckpt/`, you can test our models on example inputs using
```
python demo.py frame1.png frame2.png --model recover_cx --display
```
To change the model you can replace `recover_cx` by `recover_rn` or `recover_mn` and to also save the output you can add the argument `--out out.png`.

## Measuring Compute Resources
The script `measure.py` can be used to measure the necessary compute operations, memory, and inference time for a specific resolution:
```
python measure.py --model <MODEL> <HEIGHT> <WIDTH>
```
e.g.,
```
python measure.py --model recover_cx 1920 1080
```

## Training
Training ReCoVEr involves four different stages where each stage uses a different dataset.
To reproduce our training results you have to run the following commands:
```
python train.py --cfg config/recover_rn/Tartan.json
python train.py --cfg config/recover_rn/Tartan-C.json
python train.py --cfg config/recover_rn/Tartan-C-T-cutoff.json
python train.py --cfg config/recover_rn/Tartan-C-T-TSKH-cutoff.json
```
You might have to replace the line `"restore_ckpt": "checkpoints/Tartan-RN_exp1.pth",` in the config files by the actual name of your checkpoints after each training stage. The training for ReCoVEr-MN and ReCoVEr-CX can be reproduced in the same way by just replacing `recover_rn` in the config path by the corresponding model.


## Citation
If you find our work helpful consider citing the following paper and ⭐ this repository.
```
@inproceedings{Kiefhaber:2025:recover,
    title     = {Removing Cost Volumes from Optical Flow Estimators},
    author    = {Simon Kiefhaber and Stefan Roth and Simone Schaub-Meyer},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025}
}
```

## Acknowledgments
This repository is based on the [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) code. We thank [Yihan Wang](https://memoryslices.github.io/) for open-sourcing it.

