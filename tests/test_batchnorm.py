import torch
import torch_blocksparse
from time import time
from utils import *
from nose.tools import nottest
from parameterized import parameterized

@nottest
def run_batchnorm2d_reference(x, dy, weight, bias, eps, momentum):
  N, C, H, W = x.shape
  batchnorm2d = torch.nn.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(x.device)
  batchnorm2d.weight.data.copy_(weight)
  batchnorm2d.bias.data.copy_(bias)
  y = batchnorm2d(x)
  y.backward(dy)
  dx = x.grad.clone()
  return y, dx, None, None

@nottest
def run_batchnorm2d_triton(x, dy, weight, bias, eps, momentum):
  N, C, H, W = x.shape
  x = torch_blocksparse.Conv2d.nchw_to_chwn(x)
  dy = torch_blocksparse.Conv2d.nchw_to_chwn(dy)
  x.retain_grad()
  batchnorm2d = torch_blocksparse.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(x.device)
  batchnorm2d.weight.data.copy_(weight)
  batchnorm2d.bias.data.copy_(bias)
  y = batchnorm2d(x)
  y.backward(dy)
  dx = x.grad.clone()
  return y, dx, None, None

@nottest
def init_input(N, C, H, W, dtype):
  x = torch.rand((N, C, H, W), requires_grad=True).cuda().type(dtype)
  x.retain_grad()
  dy = torch.rand_like(x)
  weight = torch.rand(C, device=x.device, dtype=dtype)
  bias   = torch.rand(C, device=x.device, dtype=dtype)
  return x, dy, weight, bias


@nottest
def run_test_batchnorm(N, C, H, W, dtype):
  # initialize tensors
  x, dy, weight, bias = init_input(N, C, H, W, dtype)
  eps = 1e-5
  momentum = 0.1
  # execute
  ry, rdx, ry_time, rdx_time = run_batchnorm2d_reference(x, dy, weight, bias, eps, momentum)
  ty, tdx, ty_time, tdx_time = run_batchnorm2d_triton(x, dy, weight, bias, eps, momentum)
  # errors
  ac_y = allclose(ry, ty)
  ac_dx = allclose(rdx, tdx)
  return ac_y, ac_dx

@nottest
def run_bench_batchnorm(N, C, H, W, dtype, repeat=10):
  # initialize tensors
  rx, rdy, weight, bias = init_input(N, C, H, W, dtype)
  tx = rx.clone().as_strided((N, C, H, W), (1, H*W*N, W*N, N))
  tdy = rdy.clone().as_strided((N, C, H, W), (1, H*W*N, W*N, N))
  eps = 1e-5
  momentum = 0.1
  # benchmark reference
  batchnorm2d = torch.nn.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(rx.device)
  y = batchnorm2d(rx)
  backward = y.grad_fn
  time_ry = bench(lambda: batchnorm2d(rx), repeat)
  time_rdx = bench(lambda: backward(rdy), repeat)
  # benchmark triton
  batchnorm2d = torch_blocksparse.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(tx.device)
  ty = batchnorm2d(tx)
  backward = ty.grad_fn.apply
  time_ty = bench(lambda: batchnorm2d(tx), repeat)
  time_tdx = bench(lambda: backward(tdy), repeat)
  return time_ry, time_rdx, time_ty, time_tdx


def test_op():
  ac_y, ac_dx = run_test_batchnorm(256, 32, 15, 15, torch.float32) 
  assert ac_y
  assert ac_dx

def bench_op():
  import matplotlib.pyplot as plt
  f = plt.figure(figsize=(8,8))
  time_ry,  time_ty  = [], []
  time_rdx, time_tdx = [], []
  # increasing resolution
  resol = [4, 8, 10, 12, 16, 32, 48]
  for HW in resol:
    ry, rdx, ty, tdx = run_bench_batchnorm(32, 128, HW, HW, torch.float32)
    time_ry.append(ry)
    time_ty.append(ty)
    time_rdx.append(rdx)
    time_tdx.append(tdx)
  ax = f.add_subplot(221)
  ax.plot(resol, time_ry, label='Torch (NCHW)')
  ax.plot(resol, time_ty, label='Triton (CHWN)')
  ax.legend()
  ax = f.add_subplot(222)
  ax.plot(resol, time_rdx, label='Torch (NCHW)')
  ax.plot(resol, time_tdx, label='Triton (CHWN)')
  ax.legend()
  # increasing channels
  chans = [16, 32, 64, 128, 256, 512]
  time_ry,  time_ty  = [], []
  time_rdx, time_tdx = [], []
  for C in chans:
    ry, rdx, ty, tdx = run_bench_batchnorm(32, C, 15, 15, torch.float32)
    time_ry.append(ry)
    time_ty.append(ty)
    time_rdx.append(rdx)
    time_tdx.append(tdx)
  ax = f.add_subplot(223)
  ax.plot(chans, time_ry, label='Torch (NCHW)')
  ax.plot(chans, time_ty, label='Triton (CHWN)')
  ax.legend()
  ax = f.add_subplot(224)
  ax.plot(chans, time_rdx, label='Torch (NCHW)')
  ax.plot(chans, time_tdx, label='Triton (CHWN)')
  ax.legend()
  f.suptitle('Batch Normalization Performance')
  plt.show()
  