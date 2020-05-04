# Torch-Blocksparse

Block-sparse operations for PyTorch

## Current State

The following functions are supported:
```
Matrix Multiplication: SPARSE = op(DENSE) x op(DENSE)
Matrix Multiplication: DENSE = op(SPARSE) x op(DENSE)
Matrix Multiplication: DENSE = op(DENSE) x op(SPARSE)
Softmax: Sparse = Softmax(Sparse)
```
where `op()` is identity or transposition.

The following modules are supported:
```
Sparse MultiHead Attention (https://arxiv.org/abs/1904.10509)
Linear
Block-Sparse Convolutions
```

Inputs are FP32 or FP16 (with tensor cores).

## Installation

Torch-Blocksparse depends on CUDA 10.1 and the [Triton](https://github.com/ptillet/triton) language and compiler:
```
sudo apt-get install llvm-8-dev;
pip install -r requirements.txt;
python setup.py install;
```

And run the tests:
```
cd tests;
python test.py
```
The first run will take some time as all the necessary CUDA code will be JIT-compiled and cached in `$HOME/.triton/cache`.


## Usage

```python
import torch
import torch_blocksparse

# Z: non-sparse batch dimension
# H: sparse batch dimension
# M: row dimension
# N: column dimension
Z, H, M, N, K = 4, 2, 256, 512, 384
a = torch.rand((Z, H, M, K), dtype=torch.float32).cuda()
b = torch.rand((Z, H, K, N), dtype=torch.float32).cuda()
# create sparsity layout
block = 16
layout = torch.randint(0, 2, (H, M//block, N//block))
# create object for Sparse = trans(Dense) x Dense (sdd)
# some overhead there as it pre-computes look-up tables 
# internally needed by GPU kernels
dot = torch_blocksparse.SparseMatMul(layout, block, 'sdd', trans_a=True, trans_b=False)
c = dot(a, b)
# create object for Sparse = softmax(Sparse)
softmax = torch_blocksparse.SparseSoftmax(layout, block)
d = softmax(c)
```

## Performance

Here is the performance of this package compared to OpenAI blocksparse for the DDS layout (dense = dense x sparse) with square, non-transposed inputs:

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=717347395&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=1552535399&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=399094259&format=image)



The file `test.py` includes simple benchmarking code.