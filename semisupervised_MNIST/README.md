# Superpixel datasets experiments
## 1. Requirements
Please follow this [repo](https://github.com/graphdeeplearning/benchmarking-gnns) to create your environment and download datasets.

## 2. Pre-training:
`python main_superpixels_rgcl.py `

## 3. Finetuning:
`python main_superpixels_graph_classification.py --model_file /rgcl_gnn.pkl --rg_file /rgcl_rationale_generator.pkl`
