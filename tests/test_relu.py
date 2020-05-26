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
def run_test_relu(N, C, H, W, dtype):
  device = 'cuda'
  x = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  res = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  dy = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  bias = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  scale = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  # execute
  ry, rdx, rdres, rdbias, rdscale = run_relu_reference(x, res, bias, scale, dy)
  ty, tdx, tdres, tdbias, tdscale = run_relu_triton(x, res, bias, scale, dy)
  return allclose(ry, ty),\
         allclose(rdx, tdx),\
         allclose(rdres, tdres),\
         allclose(rdbias, tdbias),\
         allclose(rdscale, tdscale)

def test_full_fp32():
  ac_y, ac_dx, ac_res, ac_bias, ac_scale = run_test_relu(32, 256, 15, 15, torch.float32)
  assert ac_y
  assert ac_dx
  assert ac_res
  assert ac_bias
  assert ac_scale