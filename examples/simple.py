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