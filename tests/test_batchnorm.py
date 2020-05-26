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
def run_test_batchnorm(N, C, H, W, dtype):
  # initialize tensors
  x = torch.rand((N, C, H, W), requires_grad=True).cuda().type(dtype)
  x.retain_grad()
  dy = torch.rand_like(x)
  weight = torch.rand(C, device=x.device, dtype=dtype)
  bias   = torch.rand(C, device=x.device, dtype=dtype)
  eps = 1e-5
  momentum = 0.1
  # execute
  ry, rdx, ry_time, rdx_time = run_batchnorm2d_reference(x, dy, weight, bias, eps, momentum)
  ty, tdx, ty_time, tdx_time = run_batchnorm2d_triton(x, dy, weight, bias, eps, momentum)
  # errors
  ac_y = allclose(ry, ty)
  ac_dx = allclose(rdx, tdx)
  return ac_y, ac_dx


def test_full_fp32():
  ac_y, ac_dx = run_test_batchnorm(256, 32, 15, 15, torch.float32) 
  assert ac_y
  assert ac_dx