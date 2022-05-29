for dataset in NCI1 PROTEINS DD MUTAG COLLAB REDDIT-BINARY
do
for seed in 0 1 2 3 4 5 6 7 8 9
do
python rgcl.py --seed $seed --DS $dataset
done
done
