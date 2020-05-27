import torch
import torch_blocksparse
from time import time
from collections import OrderedDict
from utils import *
from nose.tools import nottest
from parameterized import parameterized

@nottest
def init_inputs(N, C, H, W, in_order, out_order, dtype):
  shape  = (N, C, H, W)
  stride_x = torch_blocksparse._permute.strides(N, C, H, W, in_order)
  stride_y = torch_blocksparse._permute.strides(N, C, H, W, out_order)
  x = torch.rand(N*C*H*W, requires_grad=True).as_strided(shape, stride_x).cuda().type(dtype)
  y = torch.empty_strided(shape, stride_y, device=x.device, dtype=dtype)
  return x, y

@nottest
def run_test_permute(N, C, H, W, in_order, out_order, dtype):
  x, ry = init_inputs(N, C, H, W, in_order, out_order, dtype)
  ry.copy_(x)
  ty = torch_blocksparse._permute.apply(x, in_order, out_order)
  ac_y = allclose(ry, ty)
  return ac_y


@nottest
def run_bench_permute(N, C, H, W, in_order, out_order, dtype, repeat=10):
  x, _ = init_inputs(N, C, H, W, in_order, out_order, dtype)
  fn   = torch_blocksparse._permute.apply
  time = bench(lambda: fn(x, in_order, out_order), repeat)
  gb   = 2*nbytes(x)*1e-9
  return time, gb

@parameterized(
  [
    (dtype, in_fmt, out_fmt) for dtype in [torch.float16, torch.float32]\
                             for in_fmt in ['NCHW', 'CHWN']\
                             for out_fmt in ['NCHW', 'CHWN']\
                             if in_fmt != out_fmt
  ]
)
def test_op(dtype, in_fmt, out_fmt):
  ac_y = run_test_permute(32, 32, 4, 4, in_fmt, out_fmt, dtype)
  assert ac_y

def bench_op():
  import matplotlib.pyplot as plt
  resol = [4, 8, 10, 12, 16, 32, 48, 64, 80, 96]
  f = plt.figure()
  ax = f.add_subplot(111)
  for src in ['CHWN', 'NCHW']:
    for dst in ['CHWN', 'NCHW']:
      if src == dst:
        continue
      gbps = []
      for HW in resol:
        time, gb = run_bench_permute(32, 32, HW, HW, src, dst, torch.float32)
        gbps.append(gb/time)
      ax.plot(resol, gbps, label=f'{src} -> {dst}')
      plt.legend()
  plt.show()

