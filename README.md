

<img src="figures/log.png" title="Logo" width="300" /> 

## Update

* 2020.03.13: Release tensorflow-version and pytorch-version DBG complete code.
* 2019.11.12: Release tensorflow-version DBG inference code.
* 2019.11.11: DBG is accepted by AAAI2020.
* 2019.11.08: Our ensemble DBG ranks No.1 on [ActivityNet](http://activity-net.org/challenges/2019/evaluation.html) 

## Introduction
In this repo, we propose a novel and unified action detection framework, named DBG, with superior performance over the state-of-the-art action detectors [BSN](https://arxiv.org/abs/1806.02964) and [BMN](https://arxiv.org/abs/1907.09702). You can use the code to evaluate our DBG for action proposal generation or action detection. For more details, please refer to our paper [Fast Learning of Temporal Action Proposal via Dense Boundary Generator](https://arxiv.org/pdf/1911.04127.pdf)!

## Contents

* [Paper Introduction](#paper-introduction)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
* [Citation](#citation)
* [Contact](#contact)

## Paper Introduction

 <img src="./figures/frameworkv2.PNG" width = "1000px" alt="image" align=center />

This paper introduces a novel and unified temporal action proposal generator named Dense Boundary Generator (DBG). In this work, we propose dual stream BaseNet to generate two different level and more discriminative features. We then adopt a temporal boundary classification module to predict precise temporal boundaries, and an action-aware completeness regression module to provide reliable action completeness confidence.

### ActivityNet1.3 Results
<p align='center'>
 <img src="./figures/ActivityNet.PNG" width = "800px" alt="image" align=center />
</p>

### THUMOS14 Results
<p align='center'>
 <img src="./figures/THUMOS14v3.PNG" width = "1000px" alt="image" align=center />
</p>

### Qualitative Results
<p align='center'>
  <img src='./figures/Qualitative.PNG' width=800'/>
</p>

## Prerequisites

- Tensorflow == 1.9.0 or PyTorch == 1.1
- Python == 3.6
- NVIDIA GPU == Tesla P40 
- Linux CUDA 9.0 CuDNN
- gcc 5

## Getting Started

### Installation

Clone the github repository. We will call the cloned directory as `$DBG_ROOT`.  
```bash
cd $DBG_ROOT
```
Firstly, you should compile our proposal feature generation layers. 

Please compile according to the framework you need.

Compile **tensorflow-version** proposal feature generation layers:
```bash
cd tensorflow/custom_op
make
```
Compile **pytorch-version** proposal feature generation layers:
```bash
cd pytorch/custom_op
python setup.py install
```

### Download Datasets

Prepare ActivityNet 1.3 dataset. You can use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube. Some videos have been deleted from YouTube，and you can also ask for the whole dataset by email.

Extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the 19993 rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1MYzegWXgfZd-DD9gi_GPyZ_YAN5idiFV/view?usp=sharing) or [微云](https://share.weiyun.com/5FD85UY).
Then put the features to `data/tsn_anet200` directory.

For generating the video features, scripts in `./tools` will help you to start from scrach.

### Testing of DBG

If you don't want to train the model, you can run the testing code directly using the pretrained model.
 
Pretrained model is included in `output/pretrained_model` and set parameters on `config/config_pretrained.yaml`.
Please check the `feat_dir` in `config/config_pretrained.yaml` and use scripts to run DBG.

```bash
# TensorFlow version (AUC result = 68.37%):
python tensorflow/test.py config/config_pretrained.yaml
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

# PyTorch version (AUC result = 68.26%):
python pytorch/test.py config/config_pretrained.yaml
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json
```

### Training of DBG
We also provide training code of tensorflow and pytorch version. Please check the `feat_dir` in `config/config.yaml` and follow these steps to train your model: 
#### 1. Training
```bash
# TensorFlow version:
python tensorflow/train.py config/config.yaml

# PyTorch version:
python pytorch/train.py config/config.yaml
```
#### 2. Testing
```bash
# TensorFlow version:
python tensorflow/test.py config/config.yaml

# PyTorch version:
python pytorch/test.py config/config.yaml
```

#### 3. Postprocessing
```bash
python post_processing.py output/result/ results/result_proposals.json
```

#### 4. Evaluation
```bash
python eval.py results/result_proposals.json
```

## Citation
If you find DBG useful in your research, please consider citing: 
```
@inproceedings{DBG2020arXiv,
  author    = {Chuming Lin*, Jian Li*, Yabiao Wang, Ying Tai, Donghao Luo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang, Rongrong Ji},
  title     = {Fast Learning of Temporal Action Proposal via Dense Boundary Generator},
  booktitle   = {AAAI Conference on Artificial Intelligence},
  year      = {2020},
}
```

## Contact
For any question, please file an issue or contact
```
Jian Li: swordli@tencent.com
Chuming Lin: chuminglin@tencent.com
```
