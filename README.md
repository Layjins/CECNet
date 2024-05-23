# Clustered-patch Element Connection for Few-shot Learning
This is an official implementation in PyTorch of CEC (Clustered-patch Element Connection), which is accepted by IJCAI-2023.


<p align="center">
  <img src="doc/motivation.png" width="100%"/></a>
</p>

<p align="center">
  <img src="doc/PatchCluster.png" width="100%"/></a>
</p>



## Abstract

Weak feature representation problem has influenced the performance of few-shot classification task for a long time. To alleviate this problem, recent researchers build connections between support and query instances through embedding patch features to generate discriminative representations. However, we observe that there exists semantic mismatches (foreground/ background) among these local patches, because the location and size of the target object are not fixed. What is worse, these mismatches result in unreliable similarity confidences, and complex dense connection exacerbates the problem. According to this, we propose a novel Clustered-patch Element Connection (CEC) layer to correct the mismatch problem. The CEC layer leverages Patch Cluster and Element Connection operations to collect and establish reliable connections with high similarity patch features, respectively. Moreover, we propose a CECNet, including CEC layer based attention module and distance metric. The former is utilized to generate a more discriminative representation benefiting from the global clustered-patch features, and the latter is introduced to reliably measure the similarity between pair-features. Extensive experiments demonstrate that our CECNet outperforms the state-of-the-art methods on classification benchmark. Furthermore, our CEC approach can be extended into few-shot segmentation and detection tasks, which achieves competitive performances.

<p align="center">
  <img src="doc/CECNet.png" width="100%"/></a>
</p>

<p align="center">
  <img src="doc/visual.png" width="100%"/></a>
</p>


## Repository Overview
The repository is structured as follows:
- args_mini.py/args_tiered.py: Training-related configuration files.
- test_mini_args.py/test_tiered_args.py: Testing-related configuration files.
- train.py: Start training file.
- test.py: Start testing file.
- torchFewShot/: implementation source code.
- doc/: Related documentation.


## Requirements
- Torch >= 1.6.0
- Torchvision >= 0.7.0
- Python == 3.6


## Datasets
Refer to https://github.com/blue-blue272/fewshot-CAN.
<!-- ### mini-Imagenet
For mini-Imagenet experiments, please download mini-Imagenet and put it in ./miniImagenet/data/miniImagenet
and run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on maml).

### tiered-Imagenet
For tiered-Imagenet experiments, please download tiered-Imagenet and put it in ./tieredImagenet/data/
tiered-Imagenet: https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07
https://github.com/renmengye/few-shot-ssl-public

```
mkdir -p tieredImagenet/data/tieredImagenet/data
tar -xvf tiered-imagenet.tar
mv *.pkl tieredImagenet/data/tieredImagenet/data
``` -->

## Training
python3 -m torch.distributed.launch --nproc_per_node=4 train.py

## Testing
python3 -m torch.distributed.launch --nproc_per_node=2 test.py


## Citation

If you use this code for your research, please cite our paper:
```
@article{lai2023clustered,
  title={Clustered-patch Element Connection for Few-shot Learning},
  author={Lai, Jinxiang and Yang, Siqian and Zhou, Junhong and Wu, Wenlong and Chen, Xiaochen and Liu, Jun and Gao, Bin-Bin and Wang, Chengjie},
  journal={IJCAI},
  year={2023}
}
```

## Acknowledgments

This code is based on the implementations of [**tSF: Transformer-based Semantic Filter for Few-Shot Learning**](https://github.com/Layjins/FewShotLearning-tSF).
