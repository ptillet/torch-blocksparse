import torch
import torch_blocksparse
from time import time
from utils import *
from nose.tools import nottest
from parameterized import parameterized

@nottest
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

@nottest
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

@nottest
def bench_softmax_triton(x, scale, kp_mask, attn_mask, layout, block):
  sparse_softmax = torch_blocksparse.Softmax(layout, block, bench=True)
  x = dense_to_sparse(x, layout, block)
  x = sparse_softmax(x, scale=scale, key_padding_mask=kp_mask, attn_mask=attn_mask)
  return sparse_softmax.time_y*1e-9


@nottest
def run_test_softmax(Z, H, M, N, scale, rho, block, dtype):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((H, M//block, N//block))
  x = torch.rand((Z, H, M, N), dtype=dtype, requires_grad=True, device='cuda')
  dx = torch.rand_like(x)
  bool_attn_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False, device='cuda')
  fp_attn_mask = bool_attn_mask.type(dtype)
  kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=dtype, requires_grad=False, device='cuda')
  kp_mask[kp_mask==1.] = float('-inf')
  # execute
  ry, rdx = run_softmax_reference(x, scale, dx, kp_mask, bool_attn_mask, layout, block)
  ty, tdx = run_softmax_triton(x, scale, dx, kp_mask, fp_attn_mask, layout, block)
  ac_y  = allclose(ry, ty)
  ac_dx = allclose(rdx, tdx)
  return ac_y, ac_dx
  # benchmark
  #triton_ts = bench_softmax_triton(x, scale, kp_mask, fp_attn_mask, layout, block) 
  #print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

@parameterized(
  [
    (block, dtype) for block in [16, 32, 64]\
                   for dtype in [torch.float16, torch.float32]
  ]
)
def test_op(block, dtype):
  ac_y, ac_dx = run_test_softmax(2, 4, 128, 128, 0.5, 0.4, block, dtype)
  assert ac_y
  assert ac_dx