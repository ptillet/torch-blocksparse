# Torch-Blocksparse

Block-sparse operations for PyTorch

# Supported Operations

The following features are supported:
```
Convolutions with block-sparse weights:  Layout has format [K//block, C//block, R, S]. Padding/Stride supported.
Sparse MultiHead Attention (https://arxiv.org/abs/1904.10509)
Batched Matrix Multiplication: SPARSE = op(DENSE) x op(DENSE)
Batched Matrix Multiplication: DENSE = op(SPARSE) x op(DENSE)
Batched Matrix Multiplication: DENSE = op(DENSE) x op(SPARSE)
Softmax: SPARSE = Softmax(SPARSE)
```
where `op()` is identity or transposition.

Inputs are FP32 or FP16 (with tensor cores).

## Installation
Torch-Blocksparse depends on CUDA 10.1 and the [Triton](https://github.com/ptillet/triton) language and compiler, which requires llvm-{8,9}.
```
sudo apt-get install llvm-{8,9}-dev # Ubuntu
```
You can then install the latest stable version from pip
```
pip install torch-blocksparse
```
Or the latest development version from source
```
python setup.py install;
```

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
dot = torch_blocksparse.MatMul(layout, block, 'sdd', trans_a=True, trans_b=False)
c = dot(a, b)
# create object for Sparse = softmax(Sparse)
softmax = torch_blocksparse.Softmax(layout, block)
d = softmax(c)
```

## Performance

Here is the performance of this package compared to OpenAI blocksparse for the DDS layout (dense = dense x sparse) with square, non-transposed inputs:

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=717347395&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=1552535399&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=399094259&format=image)



The file `test.py` includes simple benchmarking code.
