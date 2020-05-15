import torch
import torch_blocksparse
from time import time
from collections import OrderedDict

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

############
## UTILS  ##
############

# convert dense matrix with explicit zeros to sparse matrix
def dense_to_sparse(w, mask, block):
  Z = w.size(0)
  ret = torch.empty((Z, mask.sum(), block, block), dtype=w.dtype, device=w.device)
  nnz = mask.nonzero()
  h, i, j = nnz[:, 0], nnz[:, 1], nnz[:, 2]
  for zz in range(Z):
    for idx, (hh, ii, jj) in enumerate(zip(h, i, j)):
      ret[zz, idx, :, :] = w[zz, hh, ii*block: (ii+1)*block, jj*block: (jj+1)*block]
  return ret

# convert sparse matrix to dense matrix with explicit zeros
def sparse_to_dense(w, mask, block, zero = 0):
  maskedw = w.clone()
  for bz, wz in enumerate(range(0, w.size(0))):
    for bh, wh in enumerate(range(0, w.size(1))):
      for bi, wi in enumerate(range(0, w.size(2), block)):
        for bj, wj in enumerate(range(0, w.size(3), block)):
          if mask[bh, bi, bj] == 0:
            maskedw[wz, wh, wi : wi+block, wj:wj+block] = zero
          #maskedw[wz, wh, wi : wi+block, wj : wj+block] *= mask[bh, bi, bj]
  return maskedw

##########
# MatMul #
##########

# run reference implementation
def run_mm_reference(x, w, mode, trans_a, trans_b, layout, block, dy):
  x = sparse_to_dense(x, layout, block) if mode == 'dsd' else x
  w = sparse_to_dense(w, layout, block) if mode == 'dds' else w
  x.retain_grad()
  w.retain_grad()
  xx = x.transpose(2, 3) if trans_a else x
  ww = w.transpose(2, 3) if trans_b else w
  y = torch.matmul(xx, ww)
  y = sparse_to_dense(y, layout, block) if mode == 'sdd' else y
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  y = dense_to_sparse(y, layout, block) if mode == 'sdd' else y
  dx = dense_to_sparse(dx, layout, block) if mode == 'dsd' else dx
  dw = dense_to_sparse(dw, layout, block) if mode == 'dds' else dw
  return y, dx, dw

# run triton implementation
def run_mm_triton(x, w, mode, trans_a, trans_b, layout, block, dy):
  x = dense_to_sparse(x, layout, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, layout, block) if mode == 'dds' else w
  dy = dense_to_sparse(dy, layout, block) if mode == 'sdd' else dy
  op = torch_blocksparse.MatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
  x.retain_grad()
  w.retain_grad()
  y = op(x, w)
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  return y, dx, dw

# benchmark triton implementation
def bench_mm_triton(x, w, mode, trans_a, trans_b, layout, block, num_repeat):
  from time import time
  x = dense_to_sparse(x, layout, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, layout, block) if mode == 'dds' else w
  op = torch_blocksparse.MatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
  op.bench = num_repeat
  y = op(x, w)
  torch.cuda.synchronize()
  y = op(x, w)
  torch.cuda.synchronize()
  return op.time_c*1e-9
  
def bench_mm_openai(x, w, mode, trans_a, trans_b, layout, block, num_repeat):
  # import and disable all logging
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  import warnings
  warnings.filterwarnings('ignore',category=FutureWarning)
  from blocksparse.matmul import BlocksparseMatMul
  from blocksparse.transformer import BlocksparseTransformer
  import tensorflow as tf
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  import numpy as np
  sparsity = layout.cpu().numpy()
  # create operator
  transformer = BlocksparseTransformer(sparsity, heads=layout.shape[0], block_size=block)
  dot_sdd_nt = transformer.nt_op
  dot_dsd_tn = transformer.tn_op
  dot_dsd_nn = transformer.nn_op
  dot_dds_nn = None if mode != 'dds' else BlocksparseMatMul(sparsity[0,:,:], block_size=block)
  key = (mode, trans_a, trans_b)
  ops = {('sdd', False, True): dot_sdd_nt,
         ('dsd', True, False): dot_dsd_tn,
         ('dsd', False, False): dot_dsd_nn,
         ('dds', False, False): dot_dds_nn}
  if x.dtype == torch.float32 and (mode == 'dsd' or block != 32):
    return None
  if key not in ops:
    return None
  if mode == 'dds' and x.shape[0]*x.shape[1] != 1:
    return None
  op = ops[key]
  # placeholder
  x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
  w = w.view(w.shape[0]*w.shape[1], w.shape[2], w.shape[3])
  sparse_shape = [x.shape[0], layout.shape[0], layout[0].sum(), block, block]
  vx = tf.placeholder(tf.float32, shape = sparse_shape if mode == 'dsd' else x.shape)
  vw = tf.placeholder(tf.float32, shape = sparse_shape if mode == 'dds' else w.shape)
  x = np.random.rand(*sparse_shape) if mode == 'dsd' else x.cpu().detach().numpy()
  w = np.random.rand(*sparse_shape) if mode == 'dds' else w.cpu().detach().numpy()
  # Block-sparse matrix multiplication
  y = op(vx, vw, bench=num_repeat)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([y], feed_dict = {vx: x, vw: w})
  sess.close()

def test_mm(Z, H, M, N, K, rho, mode, trans_a, trans_b, block):
  torch.manual_seed(1)
  AS0 = K if trans_a else M
  AS1 = M if trans_a else K
  BS0 = N if trans_b else K
  BS1 = K if trans_b else N
  shape = {'sdd': (M, N),
           'dsd': (AS0, AS1),
           'dds': (BS0, BS1)}[mode]
  # initialize layout
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  layout = generator.sample((H, shape[0]//block, shape[1]//block))
  layout[:] = layout[0, :, :]
  x = torch.rand((Z, H, AS0, AS1), dtype=torch.float32, requires_grad=True).cuda()
  w = torch.rand((Z, H, BS0, BS1), dtype=torch.float32, requires_grad=True).cuda()
  dy = torch.rand((Z, H, M, N), dtype=torch.float32).cuda()
  x.retain_grad()
  w.retain_grad()
  # run
  ry, rdx, rdw = run_mm_reference(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, dy)
  ty, tdx, tdw = run_mm_triton(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, dy)
  # test
  idx = (tdx - rdx).abs() > 1
  assert(torch.allclose(ty, ry))
  assert(torch.allclose(tdx, rdx))
  assert(torch.allclose(tdw, rdw))
  # benchmark
  num_repeat = 100
  triton_ts = bench_mm_triton(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
  #openai_ts = bench_mm_openai(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
  #flops = 2 * M * bsz * bsz * layout.sum()
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

###########
# Softmax #
###########

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

###########
# CONV    #
###########

def mask_weights(w, layout, block):
  repeat_k = block*torch.ones(layout.shape[0], dtype=torch.int64)
  repeat_c = block*torch.ones(layout.shape[1], dtype=torch.int64)
  mask = layout.repeat_interleave(repeat_k, dim=0)\
               .repeat_interleave(repeat_c, dim=1).cuda().type(w.dtype)
  return w * mask

def compress_weights(w, layout, block):
  blocks = torch.empty((layout.sum(), block, block), dtype=w.dtype, device=w.device)
  current = 0
  for k in range(layout.shape[0]):
    for r in range(layout.shape[2]):
      for s in range(layout.shape[3]):
        for c in range(layout.shape[1]):
          if layout[k, c, r, s] == 0:
            continue
          blocks[current, :] = w[k*block : (k+1)*block,
                                 c*block : (c+1)*block,
                                 r, s]
          current += 1
  return blocks

def run_conv2d_reference(x, w, dy, pad, stride, layout, block, do_bench = True):
  # create conv2d
  C, K, R, S = x.shape[1], dy.shape[1], layout.shape[2], layout.shape[3]
  conv2d = torch.nn.Conv2d(w.shape[1], w.shape[0], (R, S), padding=pad, stride=stride, bias=False).cuda().type(w.dtype)
  conv2d.weight.data.copy_(mask_weights(w, layout, block))
  # run conv2d
  y = conv2d(x)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  dw = compress_weights(dw, layout, block)
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  # benchmark
  ret = [y, dx, dw, None, None, None]
  if do_bench:
    repeat = 10
    x = torch.rand_like(x)
    y = conv2d(x)
    y.backward(dy, retain_graph=True)
    # time forward
    torch.cuda.synchronize()
    y_start = time()
    for i in range(repeat):
      y = conv2d(x)
    torch.cuda.synchronize()
    ret[3] = (time() - y_start) / repeat
    # time data gradient
    conv2d.weight.requires_grad_(False)
    x.requires_grad_(True)
    torch.cuda.synchronize()
    dx_start = time()
    for i in range(repeat):
      y.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    ret[4] = (time() - dx_start) / repeat
    # time weight gradient
    conv2d.weight.requires_grad_(True)
    x.requires_grad_(False)
    torch.cuda.synchronize()
    dw_start = time()
    for i in range(repeat):
      y.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    ret[5] = (time() - dw_start) / repeat
  return tuple(ret)

def run_conv2d_triton(x, w, dy, pad, stride, layout, block, order, do_bench = True):
  # create conv2d
  N, C, H, W = x.shape
  _, _, R, S = layout.shape
  K = dy.shape[1]
  if order == 'CHWN':
    x = x.permute(1,2,3,0).contiguous().permute(3,0,1,2)
    dy = dy.permute(1,2,3,0).contiguous().permute(3,0,1,2)
    x.retain_grad()
  conv2d = torch_blocksparse.Conv2d(w.shape[1], w.shape[0], (R, S), layout, block, padding=pad, stride=stride, order=order, bias=False).cuda().type(w.dtype)
  conv2d.weight.data.copy_(compress_weights(w, layout, block))
  y = conv2d(x)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  # benchmark
  ret = [y, dx, dw, None, None, None]
  if do_bench:
    repeat = 10
    x = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    y = conv2d(x)
    y.backward(dy, retain_graph=True)
    # time forward pass
    torch.cuda.synchronize()
    y_start = time()
    for i in range(repeat):
      y = conv2d(x)
    torch.cuda.synchronize()
    ret[3] = (time() - y_start) / repeat
    # time data gradient
    conv2d.weight.requires_grad_(False)
    x.requires_grad_(True)
    torch.cuda.synchronize()
    dx_start = time()
    for i in range(repeat):
      y.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    ret[4] = (time() - dx_start) / repeat
    # time weight gradient
    conv2d.weight.requires_grad_(True)
    x.requires_grad_(False)
    torch.cuda.synchronize()
    dw_start = time()
    for i in range(repeat):
      y.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    ret[5] = (time() - dw_start) / repeat
  return tuple(ret)

def relerr(x, y):
  x = x[x.abs() != 0]
  y = y[y.abs() != 0]
  if x.shape != y.shape:
    return 1
  diff  = x - y
  ewmax = torch.max(x.abs(), y.abs()).max()
  return (diff.abs().max() / ewmax).item()

def test_conv2d(N, C, H, W, K, R, S, pad, stride, rho, block, order = 'CHWN', do_bench=True):
  dtype = torch.float32
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((K//block, C//block, R, S))
  layout.view(-1)[0] = 1
  P = (H + 2*pad[0] - R)//stride[0] + 1
  Q = (W + 2*pad[1] - S)//stride[1] + 1
  x = torch.rand((N, C, H, W), requires_grad=True).cuda().type(dtype)
  w = torch.rand((K, C, R, S), requires_grad=True).cuda().type(dtype)
  dy = torch.rand((N, K, P, Q)).cuda().type(dtype)
  x.retain_grad()
  # execute
  ry, rdx, rdw, r_y_time, r_dx_time, r_dw_time = run_conv2d_reference(x, w, dy, pad, stride, layout, block, do_bench=do_bench)
  ty, tdx, tdw, t_y_time, t_dx_time, t_dw_time = run_conv2d_triton(x, w, dy, pad, stride, layout, block, order, do_bench=do_bench)
  rtol = {torch.float16: 1e-2,
          torch.float32: 1e-4}[dtype]
  #print(ry)
  #print(ty)
  assert relerr(ry, ty) < rtol
  assert relerr(rdx, tdx) < rtol
  assert relerr(rdw, tdw) < rtol
  return r_y_time, t_y_time, r_dx_time, t_dx_time, r_dw_time, t_dw_time

#############
# BatchNorm #
#############

def run_batchnorm2d_reference(x, dy, weight, bias, eps, momentum):
  N, C, H, W = x.shape
  batchnorm2d = torch.nn.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(x.device)
  batchnorm2d.weight.data.copy_(weight)
  batchnorm2d.bias.data.copy_(bias)
  y = batchnorm2d(x)
  y.backward(dy)
  dx = x.grad.clone()
  return y, dx, None, None

def run_batchnorm2d_triton(x, dy, weight, bias, eps, momentum):
  N, C, H, W = x.shape
  x = torch_blocksparse.Conv2d.nchw_to_chwn(x)
  x.retain_grad()
  batchnorm2d = torch_blocksparse.BatchNorm2d(C, eps, momentum, affine=True, track_running_stats=True).to(x.device)
  batchnorm2d.weight.data.copy_(weight)
  batchnorm2d.bias.data.copy_(bias)
  y = batchnorm2d(x)
  y.backward(dy)
  dx = x.grad.clone()
  return y, dx, None, None



def test_batchnorm(N, C, H, W):
  # initialize tensors
  dtype = torch.float32
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
  rtol = {torch.float16: 1e-2,
          torch.float32: 1e-4}[dtype]
  # print((ry - ty).abs().max())
  assert relerr(ry, ty) < rtol


################
##   permute  ##
################

def test_permute(N, C, H, W, in_order, out_order):
  dtype = torch.float32
  shape  = (N, C, H, W)
  stride_x = torch_blocksparse._permute.strides(N, C, H, W, in_order)
  stride_y = torch_blocksparse._permute.strides(N, C, H, W, out_order)
  x = torch.rand(N*C*H*W, requires_grad=True).as_strided(shape, stride_x).cuda().type(dtype)
  ry = torch.empty_strided(shape, stride_y, device=x.device, dtype=dtype)
  ry.copy_(x)
  ty = torch_blocksparse._permute.apply(x, in_order, out_order)
  print(relerr(ry, ty))

##################
## fused-conv2d ##
##################

def run_fused_conv2d_reference(x, w, biasa, biasb, dy, pad, stride, layout, block, do_bench = True):
  # create conv2d
  C, K, R, S = x.shape[1], dy.shape[1], layout.shape[2], layout.shape[3]
  conv2d = torch.nn.Conv2d(w.shape[1], w.shape[0], (R, S), padding=pad, stride=stride, bias=False).cuda().type(w.dtype)
  relu = torch.nn.ReLU(inplace=True)
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
  dbiasa, dbiasb = None, None
  if biasa is not None:
    dbiasa = biasa.grad.clone()
    biasa.grad.zero_()
  if biasb is not None:
    dbiasb = biasb.grad.clone()
    biasb.grad.zero_()
  # reset gradients
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  # done
  return y, dx, dw, dbiasa, dbiasb

def run_fused_conv2d_triton(x, w, biasa, biasb, dy, pad, stride, layout, block, order = 'CHWN', do_bench = True):
  # create conv2d
  N, C, H, W = x.shape
  _, _, R, S = layout.shape
  K = dy.shape[1]
  if order == 'CHWN':
    x = x.permute(1,2,3,0).contiguous().permute(3,0,1,2)
    dy = dy.permute(1,2,3,0).contiguous().permute(3,0,1,2)
    x.retain_grad()
  conv2d = torch_blocksparse.Conv2d(w.shape[1], w.shape[0], (R, S), layout, block, padding=pad, stride=stride, order=order, bias=False).cuda().type(w.dtype)
  conv2d.weight.data.copy_(compress_weights(w, layout, block))
  relu = torch.nn.ReLU(inplace=True)
  # run conv2d
  y = conv2d(x, biasa=biasa, biasb=biasb)
  # backward
  y.backward(dy)
  dx = x.grad.clone()
  dw = conv2d.weight.grad.clone()
  dbiasa, dbiasb = None, None
  if biasa is not None:
    dbiasa = biasa.grad.clone()
    biasa.grad.zero_()
  if biasb is not None:
    dbiasb = biasb.grad.clone()
    biasb.grad.zero_()
  # reset gradients
  x.grad.zero_()
  conv2d.weight.grad.zero_()
  # done
  return y, dx, dw, dbiasa, dbiasb

def test_fused_conv2d(N, C, H, W, K, R, S, pad, stride, rho, block, do_biasa, do_biasb, order = 'CHWN', do_bench=True):
  dtype = torch.float32
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((K//block, C//block, R, S))
  layout.view(-1)[0] = 1
  P = (H + 2*pad[0] - R)//stride[0] + 1
  Q = (W + 2*pad[1] - S)//stride[1] + 1
  device = 'cuda'
  x = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  w = torch.randn((K, C, R, S), requires_grad=True, device=device, dtype=dtype)
  dy = torch.randn((N, K, P, Q), requires_grad=False, device=device, dtype=dtype)
  biasa, biasb = None, None
  if do_biasa:
    biasa = torch.randn((1,), requires_grad=True, device=x.device, dtype=x.dtype)
  if do_biasb:
    biasb = torch.randn((1,), requires_grad=True, device=x.device, dtype=x.dtype)
  # execute
  ry, rdx, rdw, rdbiasa, rdbiasb = run_fused_conv2d_reference(x, w, biasa, biasb, dy, \
                                                              pad, stride, layout, block, do_bench=do_bench)
  ty, tdx, tdw, tdbiasa, tdbiasb = run_fused_conv2d_triton(x, w, biasa, biasb, dy, \
                                                           pad, stride, layout, block, order, do_bench=do_bench)
  rtol = {torch.float16: 1e-2,
          torch.float32: 1e-4}[dtype]
  assert relerr(ry, ty) < rtol
  assert relerr(rdx, tdx) < rtol
  assert relerr(rdw, tdw) < rtol
  if do_biasa:
    assert relerr(rdbiasa, tdbiasa) < rtol
  if do_biasb:
    assert relerr(rdbiasb, tdbiasb) < rtol
  #return r_y_time, t_y_time, r_dx_time, t_dx_time, r_dw_time, t_dw_time

#############
##   ReLU  ##
#############

def run_fused_relu_reference(x, res, bias, scale, dy):
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

def run_fused_relu_triton(x, res, bias, scale, dy):
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

def test_fused_relu(N, C, H, W):
  dtype = torch.float32
  device = 'cuda'
  x = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  res = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  dy = torch.randn((N, C, H, W), requires_grad=True, device=device, dtype=dtype)
  bias = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  scale = torch.randn((1,), requires_grad=True, device=device, dtype=dtype)
  # execute
  ry, rdx, rdres, rdbias, rdscale = run_fused_relu_reference(x, res, bias, scale, dy)
  ty, tdx, tdres, tdbias, tdscale = run_fused_relu_triton(x, res, bias, scale, dy)
  rtol = {torch.float16: 1e-2,
          torch.float32: 1e-4}[dtype]
  assert(relerr(ry, ty) < rtol)
  assert(relerr(rdx, tdx) < rtol)
  assert(relerr(rdres, tdres) < rtol)
  assert(relerr(rdbias, tdbias) < rtol)
  assert(relerr(rdscale, tdscale) < rtol)



#############
# Run tests #
#############

def wrn_22_2_shapes():
  return (\
    (128, 16, 32, 32, 16, 3, 3, (1, 1), (1, 1)),
    (128, 32, 32, 32, 32, 3, 3, (1, 1), (1, 1)),
    (128, 16, 32, 32, 32, 1, 1, (0, 0), (1, 1)),
    (128, 32, 32, 32, 64, 3, 3, (1, 1), (2, 2)),
    (128, 64, 16, 16, 64, 3, 3, (1, 1), (1, 1)),
    (128, 32, 32, 32, 64, 1, 1, (0, 0), (2, 2)),
    (128, 64, 16, 16, 128, 3, 3, (1, 1), (2, 2)),
    (128, 128, 8, 8, 128, 3, 3, (1, 1), (1, 1)),
    (128, 64, 16, 16, 128, 1, 1, (0, 0), (2, 2)),
    (128, 128, 8, 8, 128, 3, 3, (1, 1), (1, 1))
  )
def wrn_28_10_shapes():
  return (\
  (128, 160, 32, 32, 160, 3, 3, (1, 1), (1, 1)),
  (128, 160, 32, 32, 320, 3, 3, (1, 1), (2, 2)),
  (128, 320, 16, 16, 320, 3, 3, (1, 1), (1, 1)),
  (128, 160, 32, 32, 320, 1, 1, (0, 0), (2, 2)),
  (128, 320, 16, 16, 320, 3, 3, (1, 1), (1, 1)),
  (128, 320, 16, 16, 640, 3, 3, (1, 1), (2, 2)),
  (128, 640, 8,  8,  640, 3, 3, (1, 1), (1, 1)),
  (128, 320, 16, 16, 640, 1, 1, (0, 0), (2, 2)),
  (128, 640, 8,  8,  640, 3, 3, (1, 1), (1, 1))\
  )

def resnet_50_shapes():
  return [\
   (256, 64, 56, 56, 64, 1, 1, (0, 0), (1, 1)), 
   (256, 64, 56, 56, 64, 3, 3, (1, 1), (1, 1)), 
   (256, 64, 56, 56, 256, 1, 1, (0, 0), (1, 1)), 
   (256, 256, 56, 56, 64, 1, 1, (0, 0), (1, 1)), 
   (256, 256, 56, 56, 128, 1, 1, (0, 0), (1, 1)), 
   (256, 128, 56, 56, 128, 3, 3, (1, 1), (2, 2)), 
   (256, 128, 28, 28, 512, 1, 1, (0, 0), (1, 1)), 
   (256, 256, 56, 56, 512, 1, 1, (0, 0), (2, 2)), 
   (256, 512, 28, 28, 128, 1, 1, (0, 0), (1, 1)), 
   (256, 128, 28, 28, 128, 3, 3, (1, 1), (1, 1)), 
   (256, 512, 28, 28, 256, 1, 1, (0, 0), (1, 1)), 
   (256, 256, 28, 28, 256, 3, 3, (1, 1), (2, 2)), 
   (256, 256, 14, 14, 1024, 1, 1, (0, 0), (1, 1)), 
   (256, 512, 28, 28, 1024, 1, 1, (0, 0), (2, 2)), 
   (256, 1024, 14, 14, 256, 1, 1, (0, 0), (1, 1)), 
   (256, 256, 14, 14, 256, 3, 3, (1, 1), (1, 1)),
   (256, 1024, 14, 14, 512, 1, 1, (0, 0), (1, 1)), 
   (256, 512, 14, 14, 512, 3, 3, (1, 1), (2, 2)), 
   (256, 512, 7, 7, 2048, 1, 1, (0, 0), (1, 1)), 
   (256, 1024, 14, 14, 2048, 1, 1, (0, 0), (2, 2)), 
   (256, 2048, 7, 7, 512, 1, 1, (0, 0), (1, 1)), 
   (256, 512, 7, 7, 512, 3, 3, (1, 1), (1, 1))
   ]
  
def mobilenet_v2_shapes():
  return [\
    (256, 32, 32, 32, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 32, 32, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 32, 32, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 32, 32, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 32, 32, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 32, 32, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 16, 16, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 16, 16, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 16, 16, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 16, 16, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 16, 16, 32, 1, 1, (0, 0), (1, 1)),
    (256, 32, 16, 16, 192, 1, 1, (0, 0), (1, 1)),
    (256, 192, 8, 8, 64, 1, 1, (0, 0), (1, 1)),
    (256, 64, 8, 8, 384, 1, 1, (0, 0), (1, 1)),
    (256, 384, 8, 8, 64, 1, 1, (0, 0), (1, 1)),
    (256, 64, 8, 8, 384, 1, 1, (0, 0), (1, 1)),
    (256, 384, 8, 8, 64, 1, 1, (0, 0), (1, 1)),
    (256, 64, 8, 8, 384, 1, 1, (0, 0), (1, 1)),
    (256, 384, 8, 8, 64, 1, 1, (0, 0), (1, 1)),
    (256, 64, 8, 8, 384, 1, 1, (0, 0), (1, 1)),
    (256, 384, 8, 8, 128, 1, 1, (0, 0), (1, 1)),
    (256, 128, 8, 8, 768, 1, 1, (0, 0), (1, 1)),
    (256, 768, 8, 8, 128, 1, 1, (0, 0), (1, 1)),
    (256, 128, 8, 8, 768, 1, 1, (0, 0), (1, 1)),
    (256, 768, 8, 8, 128, 1, 1, (0, 0), (1, 1)),
    (256, 128, 8, 8, 768, 1, 1, (0, 0), (1, 1)),
    (256, 768, 4, 4, 160, 1, 1, (0, 0), (1, 1)),
    (256, 160, 4, 4, 960, 1, 1, (0, 0), (1, 1)),
    (256, 960, 4, 4, 160, 1, 1, (0, 0), (1, 1)),
    (256, 160, 4, 4, 960, 1, 1, (0, 0), (1, 1)),
    (256, 960, 4, 4, 160, 1, 1, (0, 0), (1, 1)),
    (256, 160, 4, 4, 960, 1, 1, (0, 0), (1, 1)),
    (256, 960, 4, 4, 320, 1, 1, (0, 0), (1, 1)),
    (256, 320, 4, 4, 1280, 1, 1, (0, 0), (1, 1)),  
  ]

if __name__ == '__main__':
  # test softmax
  #test_softmax(1, 12, 128, 128, 0.5, 0.4, 16)
  # # test matmul
  #for mode in ['sdd', 'dsd', 'dds']:
  #   test_mm(3, 2, 256, 512, 384, 0.5, mode, False, False, 32)
  #   test_mm(3, 2, 256, 512, 384, 0.5, mode, True, False, 32)
  #   test_mm(3, 2, 256, 512, 384, 0.5, mode, False, True, 32)
  #   test_mm(3, 2, 256, 512, 384, 0.5, mode, True, True, 32)
  test_fused_relu(32, 256, 15, 15)
  test_fused_conv2d(32, 256, 15, 15, 256, 3, 3, (0, 0), (1, 1), 0.0, 32, False, False, order='CHWN') 
  #test_conv2d(256, 256, 16, 16, 256, 1, 1, (0, 0), (1, 1), 0.0, 32, 'NCHW') 
  #test_permute(32, 32, 4, 4, 'NCHW', 'CHWN')
  #test_conv2d(256, 256, 15, 15, 256, 1, 1, (0, 0), (1, 1), 0.70, 32, 'NHWC') 
  #for (N, C, H, W, K, R, S, pad, stride) in mobilenet_v2_shapes():
  #  print(f'Testing: {N:3d}, {C:3d}, {H:3d}, {W:3d}, {K:3d}, {R}, {S}, {pad}, {stride}... ', end='')
  #  r_y_time, t_y_time, r_dx_time, t_dx_time, r_dw_time, t_dw_time = test_conv2d(N, C, H, W, K, R, S, pad, stride, 0., 32, do_bench=False)
  #  print('pass!')
  #  #print(f'Y: {t_y_time/r_y_time:2.3f} ({t_y_time:2.4f}/{r_y_time:2.4f}),' 
  #  #     f'DX: {t_dx_time/r_dx_time:2.3f} ({t_dx_time:2.4f}/{r_dx_time:2.4f}),'
  #  #     f'DW: {t_dw_time/r_dw_time:2.3f} ({t_dw_time:2.4f}/{r_dw_time:2.4f})')
