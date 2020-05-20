import torch
import torch_blocksparse
from time import time
from utils import *


def run_softmax_triton(x, scale, dx, kp_mask, attn_mask, layout, block):
  sparse_softmax = torch_blocksparse.Softmax(layout, block, bench=False)
  dx = dense_to_sparse(dx, layout, block)
  x = dense_to_sparse(x, layout, block)
  x.retain_grad()
  y = sparse_softmax(x, scale=scale, key_padding_mask=kp_mask, key_padding_mask_mode='add', attn_mask=attn_mask, attn_mask_mode='mul')
  y.backward(dx)
  dx = x.grad.clone()
  x.grad.zero_()
  return x, dx

def run_softmax_reference(x, scale, dx, kp_mask, attn_mask, layout, block):
  x = sparse_to_dense(x, layout, block, zero=float('-inf'))
  x.retain_grad()
  if kp_mask is not None:
    bcattn_mask = attn_mask[None, None, :, :] + torch.zeros_like(x)
    x[bcattn_mask == 0] = float('-inf')
    y = torch.softmax(x*scale + kp_mask[:, None, None, :], -1)
  else:
    y = torch.softmax(x*scale, -1)
  y.backward(dx)
  dx = x.grad.clone()
  dx = dense_to_sparse(dx, layout, block)
  y = dense_to_sparse(y, layout, block)
  return y, dx
  
def bench_softmax_triton(x, scale, kp_mask, attn_mask, layout, block):
  sparse_softmax = torch_blocksparse.Softmax(layout, block, bench=True)
  x = dense_to_sparse(x, layout, block)
  x = sparse_softmax(x, scale=scale, key_padding_mask=kp_mask, attn_mask=attn_mask)
  return sparse_softmax.time_y*1e-9


def test_softmax(Z, H, M, N, scale, rho, block):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((H, M//block, N//block))
  x = torch.rand((Z, H, M, N), dtype=torch.float32, requires_grad=True).cuda()
  dx = torch.rand_like(x)
  bool_attn_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False).cuda()
  fp_attn_mask = bool_attn_mask.float()
  kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=torch.float32, requires_grad=False).cuda()
  kp_mask[kp_mask==1.] = float('-inf')
  # execute
  ry, rdx = run_softmax_reference(x, scale, dx, kp_mask, bool_attn_mask, layout, block)
  ty, tdx = run_softmax_triton(x, scale, dx, kp_mask, fp_attn_mask, layout, block)
  ry = ry[ry == ry]
  ty = ty[ty == ty]
  rdx = rdx[(rdx == rdx) & (rdx != 0)]
  tdx = tdx[(tdx == tdx) & (tdx != 0)]
  assert(torch.allclose(ry, ty))
  assert(torch.allclose(rdx, tdx))
  # benchmark
  triton_ts = bench_softmax_triton(x, scale, kp_mask, fp_attn_mask, layout, block) 
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

if __name__ == '__main__':
  test_softmax(1, 12, 128, 128, 0.5, 0.4, 16)
