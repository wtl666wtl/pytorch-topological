## DSC 214

Project of DSC 214.
Folked from [pytorch topological](https://github.com/aidos-lab/pytorch-topological).

## Run Experiments

run normal GCN (4-layer GCN)

```
python GCN.py --dataset {dataset name}
```

run normal TGNN (3-layer GCN + 1 TOGL)

```
python TGNN.py --dataset {dataset name}
```

run attention-based TGNN (3-layer GCN + 1 ATOGL)

```
python ATGNN.py --dataset {dataset name}
```

## Reference
[TOPOLOGICAL GRAPH NEURAL NETWORKS](https://arxiv.org/pdf/2102.07835)
