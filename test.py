import torch_blocksparse
import torch
torch.manual_seed(0)

def reference_dot(x, w, mask):
  WS0, WS1 = w.size()
  MS0, MS1 = mask.size()
  assert WS0 % MS0 == 0
  assert WS1 % MS1 == 0
  block_size_0 = WS0 // MS0
  block_size_1 = WS1 // MS1
  assert block_size_0 == block_size_1
  maskedw = w.clone()
  for bi, wi in enumerate(range(0, WS0, block_size_0)):
    for bj, wj in enumerate(range(0, WS1, block_size_1)):
      maskedw[wi : wi+block_size_0,
              wj : wj+block_size_1] *= mask[bi, bj]
  return torch.matmul(x, maskedw)

def run_reference(x, w, mask, bsz, dy):
  y = reference_dot(x, w, mask)
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  return y, dx, dw

def run_triton(x, w, mask, bsz, dy):
  linear = torch_blocksparse.Linear(x.size(0), x.size(1), bsz, mask).cuda()
  linear.weight.data.copy_(w)
  y = linear(x)
  y.backward(dy)
  dx = x.grad.clone()
  dw = linear.weight.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  return y, dx, dw

def bench_triton(x, bsz, mask, num_repeat):
  from time import time
  linear = torch_blocksparse.Linear(x.size(0), x.size(1), bsz, mask).cuda()
  # benchmark forward pass
  y = linear(x)
  start = time()
  for i in range(num_repeat):
    y = linear(x)
  end = time()
  ty = (end - start) / num_repeat
  return ty
  
# parameters
M, N, K = 256, 256, 256
bsz = 16
# initialize inputs
mask = torch.randint(0, 2, (K//bsz, N//bsz))
x = torch.rand((M, K), dtype=torch.float32, requires_grad=True).cuda()
w = torch.rand((K, N), dtype=torch.float32, requires_grad=True).cuda()
dy = torch.rand((M, N), dtype=torch.float32).cuda()
x.retain_grad()
w.retain_grad()
# run
ry, rdx, rdw = run_reference(x, w, mask, bsz, dy)
ty, tdx, tdw = run_triton(x, w, mask, bsz, dy)
# test
assert(torch.allclose(ty, ry))
assert(torch.allclose(tdx, rdx))
assert(torch.allclose(tdw, rdw))
# benchmark
num_repeat = 10
ty = bench_triton(x, bsz, mask, num_repeat)
print(ty)