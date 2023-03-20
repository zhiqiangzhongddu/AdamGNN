# Multi-grained Semantics-aware Graph Neural Networks

Implementation of the AdamGNN with Pytorch, another implementation with Tensorflow incoming.

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
Two demo file is given to show the execution of link prediction (LP) and node classification (NC) tasks.

## Citation

Please cite our paper if you make use of this code in your own work:

```bibtex
@article{ZLP221,
author = {Zhiqiang Zhong and Cheng{-}Te Li and Jun Pang},
title = {Multi-grained Semantics-aware Graph Neural Networks},
journal = {IEEE Transactions on Knowledge and Data Engineering (TKDE)},
year = {2022},
}
```
