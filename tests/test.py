from torch_blocksparse import *
import torch
torch.manual_seed(0)

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
  op = SparseMatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
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
  op = SparseMatMul(layout, block, mode, trans_a=trans_a, trans_b=trans_b)
  op.bench = num_repeat
  y = op(x, w)
  torch.cuda.synchronize()
  y = op(x, w)
  torch.cuda.synchronize()
  return op.time_c*1e-9
  
def bench_mm_openai(x, mode, trans_a, trans_b, layout, block, num_repeat):
  # import and disable all logging
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  import warnings
  warnings.filterwarnings('ignore',category=FutureWarning)
  from blocksparse.matmul import BlocksparseMatMul, BlocksparseTransformer
  import tensorflow as tf
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  import numpy as np
  sparsity = layout.cpu().numpy()
  # create operator
  transformer = BlocksparseTransformer(sparsity, block_size=block)
  dot_sdd_nt = transformer.nt_op
  dot_dsd_tn = transformer.tn_op
  dot_dsd_nn = transformer.nn_op
  dot_dds_nn = BlocksparseMatMul(sparsity, block_size=block)
  key = (mode, trans_a, trans_b)
  ops = {('sdd', False, True): dot_sdd_nt,
         ('dsd', True, False): dot_dsd_tn,
         ('dsd', False, False): dot_dsd_nn,
         ('dds', False, False): dot_dds_nn}
  if key not in ops:
    return None
  # shapes
  M, K = x.size()
  N = layout.size(1)*block
  # placeholder
  vx = tf.placeholder(tf.float32, shape=[None, K])
  vw = tf.placeholder(tf.float32, shape=dot.w_shape)
  w = np.random.rand(*dot.w_shape)
  # Block-sparse matrix multiplication
  y = dot_dds_nn(vx, vw, bench=num_repeat)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([y], feed_dict = {vx: x.cpu().detach().numpy(), vw: w})
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
  #openai_ts = bench_mm_openai(x, bsz, layout, num_repeat)
  #flops = 2 * M * bsz * bsz * layout.nonzero().size(0)
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

###########
# Softmax #
###########

def run_softmax_triton(x, scale, dx, mask, layout, block):
  sparse_softmax = softmax.SparseSoftmax(layout, block, bench=False)
  dx = dense_to_sparse(dx, layout, block)
  x = dense_to_sparse(x, layout, block)
  x.retain_grad()
  y = sparse_softmax(x, scale=scale, mask=mask)
  y.backward(dx)
  dx = x.grad.clone()
  x.grad.zero_()
  return x, dx

def run_softmax_reference(x, scale, dx, mask, layout, block):
  x = sparse_to_dense(x, layout, block, zero=float('-inf'))
  x.retain_grad()
  if mask is not None:
    y = torch.softmax(x*scale + mask[:, None, None, :], -1)
  else:
    y = torch.softmax(x*scale, -1)
  y.backward(dx)
  dx = x.grad.clone()
  dx = dense_to_sparse(dx, layout, block)
  y = dense_to_sparse(y, layout, block)
  return y, dx
  
def bench_softmax_triton(x, scale, mask, layout, block):
  sparse_softmax = softmax.SparseSoftmax(layout, block, bench=True)
  x = dense_to_sparse(x, layout, block)
  x = sparse_softmax(x, scale=scale, mask=mask)
  return sparse_softmax.time_y*1e-9


def test_softmax(Z, H, M, N, scale, rho, block):
  # probability distribution
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  # initialize tensors
  layout = generator.sample((H, M//block, N//block))
  x = torch.rand((Z, H, M, N), dtype=torch.float32, requires_grad=True).cuda()
  dx = torch.rand_like(x)
  mask = torch.randint(low=0, high=1, size=(Z, N), dtype=torch.float32, requires_grad=False).cuda()
  mask[mask==1.] = float('-inf')
  # execute
  ry, rdx = run_softmax_reference(x, scale, dx, mask, layout, block)
  ty, tdx = run_softmax_triton(x, scale, dx, mask, layout, block)
  assert(torch.allclose(ry, ty))
  assert(torch.allclose(rdx, tdx))
  # benchmark
  triton_ts = bench_softmax_triton(x, scale, mask, layout, block) 
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')

#############
# Run tests #
#############

if __name__ == '__main__':
  # test softmax
  test_softmax(3, 2, 256, 2048, 0.5, 0.7, 16)
  # test matmul
  for mode in ['sdd', 'dsd', 'dds']:
    test_mm(2, 1, 256, 512, 384, 0.5, mode, False, False, 16)
    test_mm(2, 1, 256, 512, 384, 0.5, mode, True, False, 16)
    test_mm(2, 1, 256, 512, 384, 0.5, mode, False, True, 16)
    test_mm(2, 1, 256, 512, 384, 0.5, mode, True, True, 16)