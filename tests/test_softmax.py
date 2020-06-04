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
def init_inputs(Z, H, M, N, scale, rho, block, dtype, dense_x = True, layout = None):
  if layout is None:
    layout = make_layout(rho, (H, M//block, N//block))
  if dense_x:
    x = torch.rand((Z, H, M, N), dtype=dtype, requires_grad=True, device='cuda')
  else:
    x = torch.rand((Z, layout.sum(), block, block), dtype=dtype, requires_grad=True, device='cuda')
  dx = torch.rand_like(x)
  bool_attn_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False, device='cuda')
  fp_attn_mask = bool_attn_mask.type(dtype)
  kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=dtype, requires_grad=False, device='cuda')
  kp_mask[kp_mask==1.] = float('-inf')
  return layout, x, dx, bool_attn_mask, fp_attn_mask, kp_mask

@nottest
def run_test_softmax(Z, H, M, N, scale, rho, block, dtype, layout = None):
  layout, x, dx, bool_attn_mask, fp_attn_mask, kp_mask = init_inputs(Z, H, M, N, scale, rho, block, dtype, layout=layout)
  ry, rdx = run_softmax_reference(x, scale, dx, kp_mask, bool_attn_mask, layout, block)
  ty, tdx = run_softmax_triton(x, scale, dx, kp_mask, fp_attn_mask, layout, block)
  ac_y  = allclose(ry, ty)
  ac_dx = allclose(rdx, tdx)
  return ac_y, ac_dx

@nottest
def run_bench_softmax(Z, H, M, N, scale, rho, block, dtype, layout = None, repeat=10):
  layout, x, dx, _, attn_mask, kp_mask = init_inputs(Z, H, M, N, scale, rho, block, dtype, dense_x=False, layout=layout)
  x = x.clone()
  dx = dx.clone()
  # forward function
  sparse_softmax = torch_blocksparse.Softmax(layout, block, bench=False)
  y = sparse_softmax(x, scale, None, None, 'add', 'mul')
  # backward function
  backward = y.grad_fn.apply
  backward(dx)
  x = x.clone()
  # benchmark
  time_y  = bench(lambda: sparse_softmax(x, scale, None, None, 'add', 'mul'), repeat)
  time_dx = bench(lambda: backward(dx), repeat)
  gb_y  = (2*nbytes(x) + nbytes(attn_mask) + nbytes(kp_mask))*1e-9
  gb_dx = 3*nbytes(x)*1e-9
  return time_y, time_dx, gb_y, gb_dx

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

def bench_op():
  L = ['Forward', 'Backward']
  # Vary number of columns
  xs = [256*i for i in range(1,13)]
  ys = torch.empty((len(xs), len(L)))
  for i, N in enumerate(xs):
    time_y, time_dx, gb_y, gb_dx = run_bench_softmax(2, 4, 256, N, 0.4, 0., 16, torch.float16)
    ys[i, 0] = gb_y  / time_y
    ys[i, 1] = gb_dx / time_dx
  prettyprint(xs, ys, L, '# Columns')
  # Vary number of rows
  xs = [32*i for i in range(1,16)]
  ys = torch.empty((len(xs), len(L)))
  for i, M in enumerate(xs):
    time_y, time_dx, gb_y, gb_dx = run_bench_softmax(2, 4, M, 512, 0.4, 0., 16, torch.float16)
    ys[i, 0] = gb_y  / time_y
    ys[i, 1] = gb_dx / time_dx
  prettyprint(xs, ys, L, '# Rows')
  # Sparse GPT2
  batch, heads = 1, 1
  block, stride, nv, vs = 64, 64, 4, 1
  xs = [1024, 2048, 4096, 8192]
  ys = torch.empty((len(xs), len(L)))
  for i, ctx in enumerate(xs):
    layout = torch_blocksparse.MultiheadAttention._make_layout(heads, ctx//block, 'fixed', stride//block, True, 4, 1)
    time_y, time_dx, gb_y, gb_dx = run_bench_softmax(batch, heads, ctx, ctx, 0., 0., block, torch.float16, layout=layout)
    ys[i, 0] = gb_y  / time_y
    ys[i, 1] = gb_dx / time_dx
  prettyprint(xs, ys, L, 'Seq. Length')
  # Sparse BERT
  ys = torch.empty((len(xs), len(L)))
  for i, ctx in enumerate(xs):
    layout = torch_blocksparse.MultiheadAttention._make_layout(heads, ctx//block, 'fixed', stride//block, False, 4, 1)
    time_y, time_dx, gb_y, gb_dx = run_bench_softmax(batch, heads, ctx, ctx, 0., 0., block, torch.float16, layout=layout)
    ys[i, 0] = gb_y  / time_y
    ys[i, 1] = gb_dx / time_dx
  prettyprint(xs, ys, L, 'Seq. Length')


#bench_op()
