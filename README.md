# Let Invariant Discovery Inspire Graph Contrastive Learning

This is our PyTorch implementation for the paper:

>Sihang Li, Xiang Wang*, An Zhang, Ying-Xin Wu, Xiangnan He and Tat-Seng Chua (2022). Let Invariant Rationale Discovery Inspire Graph Contrastive Learning, [Paper in arXiv](https://arxiv.org/abs/2206.07869). In ICML'22, Baltimore, Maryland, USA, July 17-23, 2022.

Author: Sihang Li (sihang0520 at gmail.com)



## Introduction

Without supervision signals, **<u>R</u>ationale-aware <u>G</u>raph <u>C</u>ontrastive <u>L</u>earning (RGCL)** uses a rationale generator to reveal salient features about graph instance-discrimination as the rationale, and then creates rationale-aware views for contrastive learning. This rationale-aware pre-training scheme endows the backbone model with the powerful representation ability, further facilitating the fine-tuning on downstream tasks.



## Citation 

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{RGCL,
  author    = {Sihang Li and
               Xiang Wang and
               An Zhang and
               Xiangnan He and
               Tat-Seng Chua},
  title     = {Let Invariant Rationale Discovery Inspire Graph Contrastive Learning},
  booktitle = {{ICML}},
  year      = {2022}
}

```



## Experiments

* Transfer Learning on MoleculeNet datasets
* Semi-supervised learning on Superpixel MNIST dataset
* Unsupervised representation learning on TU datasets



## Potential Issues

Some issues might occur due to the version mismatch.
* ```KeyError:'num_nodes'``` in unsupervised_TU: https://github.com/Shen-Lab/GraphCL/issues/36, https://github.com/Shen-Lab/GraphCL/issues/41
* ```AttributeError: 'Data' object has no attribute 'cat_dim'``` in transferLearning_MoleculeNet_PPI: https://github.com/Shen-Lab/GraphCL/issues/13




## Acknowledgements

The backbone implementation is reference to https://github.com/Shen-Lab/GraphCL.



