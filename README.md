# Adaptive HIerarchical Graph Neural Network

Implementation of the AdaHGNN with Pytorch, another implementation with Tensorflow incoming.

### Required packages
The code has been tested running under Python 3.7.1. with the following packages installed (along with their dependencies):

- numpy == 1.18.1
- pandas == 1.0.3
- scikit-learn == 0.22.2
- networkx == 2.4
- pytorch == 1.4.0
- torch_geometric == 1.4.2

### Data requirement
All eight datasets we used in the paper are all public datasets which can be downloaded from the internet.

### Code execution
Link prediction:
```
python main.py --task LP --dataset cora --mode basemodel --model AdaHGNN --num_levels 4 --num_epoch 2001 --lr 0.0001 --relu True --dropout True --drop_ratio 0.5 --local_agg_gnn GCN --fitness_mode both_c --pooling_mode att --loss_mode kl --SEED 123 --gpu True
```

Node classification:
```
python main.py --task NC --dataset wiki --mode basemodel --model AdaHGNN --num_levels 5 --num_epoch 2001 --lr 0.0001 --relu True --dropout True --drop_ratio 0.1 --local_agg_gnn GCN --fitness_mode both_c --pooling_mode att --loss_mode all --SEED 123 --gpu True
``` 

Graph classification:
```
python main.py --task GC --dataset DD --mode basemodel --model AdaHGNN --num_levels 4 --num_epoch 2001 --lr 0.0001 --relu True --dropout True --drop_ratio 0.1 --local_agg_gnn GCN --fitness_mode both_c --pooling_mode att --loss_mode all --SEED 123 --gpu True
```
