# Graph Contrastive Learning with Augmentations

PyTorch implementation for *Let Invariant Rationale Discovery Inspire Graph Contrastive Learning*

Sihang Li, Xiang Wang*, An Zhang, Ying-Xin Wu, Xiangnan He and Tat-Seng Chua

In NeurIPS 2020.



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
