import torch
import torch_blocksparse
from time import time
from utils import *
from nose.tools import nottest
from parameterized import parameterized

@nottest
def run_relu_reference(x, res, bias, scale, dy):
  relu = torch.nn.ReLU()
  y = relu(x*scale + bias + res)
  y.backward(dy)
  # save gradients
  dx = x.grad.clone()
  dres = res.grad.clone()
  dbias = bias.grad.clone()
  dscale = scale.grad.clone()
  # reset gradients
  x.grad.zero_()
  res.grad.zero_()
  bias.grad.zero_()
  scale.grad.zero_()
  return y, dx, res, dbias, dscale

@nottest
def run_relu_triton(x, res, bias, scale, dy):
  relu = torch_blocksparse.ReLU()
  y = relu(x, scale, bias, res)
  y.backward(dy)
  # save gradients
  dx = x.grad.clone()
  dres = res.grad.clone()
  dbias = bias.grad.clone()
  dscale = scale.grad.clone()
  # reset gradients
  x.grad.zero_()
  res.grad.zero_()
  bias.grad.zero_()
  scale.grad.zero_()
  return y, dx, res, dbias, dscale

@nottest
def init_inputs(N, C, H, W, device, dtype):
  x = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  res = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  dy = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  bias = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  scale = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  return x, res, dy, bias, scale

@nottest
def run_test_relu(N, C, H, W, dtype):
  x, res, dy, bias, scale = init_inputs(N, C, H, W, 'cuda', dtype)
  ry, rdx, rdres, rdbias, rdscale = run_relu_reference(x, res, bias, scale, dy)
  ty, tdx, tdres, tdbias, tdscale = run_relu_triton(x, res, bias, scale, dy)
  return allclose(ry, ty),\
         allclose(rdx, tdx),\
         allclose(rdres, tdres),\
         allclose(rdbias, tdbias),\
         allclose(rdscale, tdscale)

@nottest
def run_bench_relu(N, C, H, W, dtype, repeat=10):
  x, res, dy, bias, scale = init_inputs(N, C, H, W, 'cuda', dtype)
  forward = torch_blocksparse.ReLU()
  y = forward(x, scale, bias, res)
  backward = y.grad_fn.apply
  time_y  = bench(lambda: forward(x, scale, bias, res), repeat)
  time_dx = bench(lambda: backward(dy), repeat)
  y.backward(dy)
  gb_y = 2*nbytes(x)*1e-9
  gb_dx = 3*nbytes(x)*1e-9
  return time_y, time_dx, gb_y, gb_dx


@parameterized([
  (dtype,) for dtype in [torch.float32]
])
def test_op(dtype):
  ac_y, ac_dx, ac_res, ac_bias, ac_scale = run_test_relu(4, 64, 15, 15, dtype)
  assert ac_y
  assert ac_dx
  assert ac_res
  assert ac_bias
  assert ac_scale

def bench_op():
  import matplotlib.pyplot as plt
  resol = [4, 8, 10, 12, 16, 32, 48, 64, 80, 96]
  f = plt.figure()
  ax = f.add_subplot(111)
  perf_y, perf_dx = [], []
  for HW in resol:
    time_y, time_dx, gb_y, gb_dx = run_bench_relu(32, 32, HW, HW, torch.float32)
    perf_y.append(gb_y/time_y)
    perf_dx.append(gb_dx/time_dx)
  ax.plot(resol, perf_y, label='Forward')
  ax.plot(resol, perf_dx, label='Backward')
  plt.legend()
  plt.show()
  