split=scaffold

for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --input_model_file models_rgcl/rgcl_gnn.pth --split $split --runseed $runseed --dataset $dataset
done
done
