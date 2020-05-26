import torch
import torch_blocksparse
from time import time
from utils import *
from nose.tools import nottest
from parameterized import parameterized

# run reference implementation
@nottest
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
@nottest
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
  
@nottest
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

@nottest
def run_test_mm(Z, H, M, N, K, rho, mode, trans_a, trans_b, block, dtype):
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
  ac_y = allclose(ry, ty)
  ac_dx = allclose(rdx, tdx)
  ac_dw = allclose(rdw, tdw)
  return ac_y, ac_dx, ac_dw
  # test
#   idx = (tdx - rdx).abs() > 1
#   assert(torch.allclose(ty, ry))
#   assert(torch.allclose(tdx, rdx))
#   assert(torch.allclose(tdw, rdw))
#   # benchmark
#   num_repeat = 100
#   triton_ts = bench_mm_triton(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
#   #openai_ts = bench_mm_openai(x.clone(), w.clone(), mode, trans_a, trans_b, layout, block, num_repeat)
#   #flops = 2 * M * bsz * bsz * layout.sum()
#   print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

@parameterized(
    [
    (mode, at, bt, 32) for mode in ['sdd', 'dsd', 'dds']\
                       for at   in [False, True]\
                       for bt   in [False, True]\
    ]\
    +\
    [
    (mode, False, False, block) for mode in ['sdd', 'dsd', 'dds']\
                                for block in [16, 32, 64]\
    ]
)
def test_op(mode, at, bt, block):
  ac_y, ac_dx, ac_dw = run_test_mm(3, 2, 256, 512, 384, 0.5, mode, 
                                  at, bt, block, torch.float32)
  assert ac_y
  assert ac_dx
  assert ac_dw