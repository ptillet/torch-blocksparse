import torch
import torch_blocksparse
from time import time
from utils import *
import unittest
from nose.tools import nottest
from parameterized import parameterized

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

@nottest
def bench_conv2d(x, w, dy, pad, stride, layout, block, repeat=10):
  # dry run
  y = conv2d(x)
  y.backward(dy, retain_graph=True)
  # time forward
  time_y = bench(lambda: conv2d(x), repeat)
  # time data gradient
  conv2d.weight.requires_grad_(False)
  x.requires_grad_(True)
  time_dx = bench(lambda: y.backward(dy, retain_graph=True))
  # time weight gradient
  conv2d.weight.requires_grad_(True)
  x.requires_grad_(False)
  time_dw = bench(lambda: y.backward(dy, retain_graph=True))
  return time_y, time_dx, time_dw

@nottest
def run_conv2d_reference(x, w, biasa, biasb, dy, pad, stride, layout, block, do_bench = False):
  # create conv2d
  C, K, R, S = x.shape[1], dy.shape[1], layout.shape[2], layout.shape[3]
  relu = torch.nn.ReLU(inplace=True)
  conv2d = torch.nn.Conv2d(C, K, (R, S), padding=pad, stride=stride, bias=False).cuda().type(w.dtype)
  conv2d.weight.data.copy_(mask_weights(w, layout, block))
  # run conv2d
  u = x
  if biasa is not None:
    u = relu(x + biasa)
  if biasb is not None:
    u = u + biasb
  y = conv2d(u)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  dw = compress_weights(dw, layout, block)
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  dbiasa, dbiasb = None, None
  if biasa is not None:
    dbiasa = biasa.grad.clone()
    biasa.grad.zero_()
  if biasb is not None:
    dbiasb = biasb.grad.clone()
    biasb.grad.zero_()
  return y, dx, dw, dbiasa, dbiasb

@nottest
def run_conv2d_triton(x, w, biasa, biasb, dy, pad, stride, layout, block, order, do_bench = False):
  # create conv2d
  N, C, H, W = x.shape
  K, R, S = dy.shape[1], layout.shape[2], layout.shape[3]
  conv2d = torch_blocksparse.Conv2d(C, K, (R, S), layout, block, padding=pad, stride=stride, order=order, bias=False).cuda()
  conv2d.weight.type(w.dtype)
  conv2d.weight.data = w
  y = conv2d(x, biasa=biasa, biasb=biasb)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  dbiasa, dbiasb = None, None
  if biasa is not None:
    dbiasa = biasa.grad.clone()
    biasa.grad.zero_()
  if biasb is not None:
    dbiasb = biasb.grad.clone()
    biasb.grad.zero_()
  return y, dx, dw, dbiasa, dbiasb

@nottest
def run_test_conv2d(N, C, H, W, K, R, S, pad, stride, rho, block, dtype, do_biasa, do_biasb, order = 'CHWN', do_bench=False):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # creates layout for testing
  layout = generator.sample((K//block, C//block, R, S))
  layout[:,1,:,:] = 0
  layout[1,:,:,:] = 0
  layout[:,:,0,0] = 0
  layout.view(-1)[0] = 1
  # initialize tensors
  P  = (H + 2*pad[0] - R)//stride[0] + 1
  Q  = (W + 2*pad[1] - S)//stride[1] + 1
  x  = torch.rand(N*C*H*W, requires_grad=True).cuda().type(dtype)
  dy = torch.rand(N*K*P*Q, requires_grad=True).cuda().type(dtype)
  rw = torch.rand((K, C, R, S), requires_grad=True).cuda().type(dtype)
  tw = compress_weights(rw, layout, block)
  biasa, biasb = None, None
  if do_biasa:
    biasa = torch.randn((1,), requires_grad=True, device=x.device, dtype=x.dtype)
  if do_biasb:
    biasb = torch.randn((1,), requires_grad=True, device=x.device, dtype=x.dtype)
  # pad memory for easier detection of out-of-bounds accesses
  x  = mempad(x,  (N, C, H, W), (1, N*W*H, N*W, N))
  dy = mempad(dy, (N, K, P, Q), (1, N*Q*P, N*Q, N))
  rw = mempad(rw, (K, C, R, S), (C*R*S, R*S, S, 1))
  tw = mempad(tw, tw.shape, tw.stride())
  # retain gradients
  x.retain_grad()
  rw.retain_grad()
  # execute
  ry, rdx, rdw, rdbiasa, rdbiasb = run_conv2d_reference(x, rw, biasa, biasb, dy, pad, stride, layout, block, do_bench=do_bench)
  ty, tdx, tdw, tdbiasa, tdbiasb = run_conv2d_triton(x, tw, biasa, biasb, dy, pad, stride, layout, block, order, do_bench=do_bench)
  # allclose ?
  ac_y = torch.allclose(ry, ty)
  ac_dx = torch.allclose(rdx, tdx)
  ac_dw = torch.allclose(rdw, tdw)
  ac_dbiasa, ac_dbiasb = None, None
  if do_biasa:
    ac_dbiasa = torch.allclose(rdbiasa, tdbiasa)
  if do_biasb:
    ac_dbiasb = torch.allclose(rdbiasb, tdbiasb)
  return ac_y, ac_dx, ac_dw, ac_dbiasa, ac_dbiasb


def test_op_fp16():
  # pass when no tensor cores
  try:
    ac_y, ac_dx, ac_dw, ac_dbiasa, ac_dbiasb = run_test_conv2d(36, 32, 27, 27, 64, 3, 3,
                                        (1, 1), (2, 2), 0.5, 16, torch.float16, True, True, 'CHWN') 
  except RuntimeError:
    return
  assert ac_y
  assert ac_dx
  assert ac_dw
  assert ac_dbiasa
  assert ac_dbiasb


def test_op_fp32():
  ac_y, ac_dx, ac_dw, ac_dbiasa, ac_dbiasb = run_test_conv2d(36, 32, 27, 27, 64, 3, 3,
                                       (1, 1), (2, 2), 0.5, 16, torch.float32, True, True, 'CHWN') 
  assert ac_y
  assert ac_dx
  assert ac_dw
  if ac_dbiasa:
    assert ac_dbiasa
  if ac_dbiasb:
    assert ac-dbiasb