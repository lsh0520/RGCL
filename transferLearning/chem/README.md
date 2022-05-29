## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue https://github.com/snap-stanford/pretrain-gnns/issues/14.
This might leads to some inconsistent results with those in the paper.



## Reproductivity

To reproduce the transfer learning results in our paper, simply run *finetune.sh*. 

We release our pre-trained model in folder *models_rgcl*.



## Training from the scratch

We suggest to run it on Linux Platform.

```
python pretrain_rgcl.py --num_workers 8
```



## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns and https://github.com/Shen-Lab/GraphCL.

