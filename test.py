import torch_blocksparse
import torch
torch.manual_seed(0)

# convert dense matrix with explicit zeros to sparse matrix
def dense_to_sparse(w, mask, block):
  ret = torch.empty((mask.sum(), block, block), dtype=w.dtype, device=w.device)
  nnz = mask.nonzero()
  i, j = nnz[:, 0], nnz[:, 1]
  for idx, (ii, jj) in enumerate(zip(i, j)):
    ret[idx, :, :] = w[ii*block: (ii+1)*block, 
                       jj*block: (jj+1)*block]
  return ret

# convert sparse matrix to dense matrix with explicit zeros
def sparse_to_dense(w, mask, block):
  maskedw = w.clone()
  for bi, wi in enumerate(range(0, w.size(0), block)):
    for bj, wj in enumerate(range(0, w.size(1), block)):
      maskedw[wi : wi+block,
              wj : wj+block] *= mask[bi, bj]
  return maskedw

# run reference implementation
def run_reference(x, w, mode, trans_a, trans_b, mask, block, dy):
  x = sparse_to_dense(x, mask, block) if mode == 'dsd' else x
  w = sparse_to_dense(w, mask, block) if mode == 'dds' else w
  x.retain_grad()
  w.retain_grad()
  xx = x.t() if trans_a else x
  ww = w.t() if trans_b else w
  y = torch.matmul(xx, ww)
  y = sparse_to_dense(y, mask, block) if mode == 'sdd' else y
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  y = dense_to_sparse(y, mask, block) if mode == 'sdd' else y
  dx = dense_to_sparse(dx, mask, block) if mode == 'dsd' else dx
  dw = dense_to_sparse(dw, mask, block) if mode == 'dds' else dw
  return y, dx, dw

# run triton implementation
def run_triton(x, w, mode, trans_a, trans_b, mask, block, dy):
  op = torch_blocksparse.SparseMatMul(mask, block, mode, trans_a=trans_a, trans_b=trans_b)
  x = dense_to_sparse(x, mask, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, mask, block) if mode == 'dds' else w
  dy = dense_to_sparse(dy, mask, block) if mode == 'sdd' else dy
  x.retain_grad()
  w.retain_grad()
  y = op(x, w)
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  return y, dx, dw

# benchmark triton implementation
def bench_triton(x, w, mode, trans_a, trans_b, mask, block, num_repeat):
  from time import time
  x = dense_to_sparse(x, mask, block) if mode == 'dsd' else x
  w = dense_to_sparse(w, mask, block) if mode == 'dds' else w
  op = torch_blocksparse.SparseMatMul(mask, block, mode, trans_a=trans_a, trans_b=trans_b)
  op.bench = num_repeat
  y = op(x, w)
  torch.cuda.synchronize()
  y = op(x, w)
  torch.cuda.synchronize()
  return op.time_c*1e-9
  
def bench_openai(x, block, mask, num_repeat):
  # import and disable all logging
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  import warnings
  warnings.filterwarnings('ignore',category=FutureWarning)
  from blocksparse.matmul import BlocksparseMatMul
  import tensorflow as tf
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  import numpy as np
  sparsity = mask.cpu().numpy()
  # create operator
  dot = BlocksparseMatMul(sparsity, block_size=block)
  # shapes
  M, K = x.size()
  N = mask.size(1)*block
  # placeholder
  vx = tf.placeholder(tf.float32, shape=[None, K])
  vw = tf.placeholder(tf.float32, shape=dot.w_shape)
  w = np.random.rand(*dot.w_shape)
  # Block-sparse matrix multiplication
  y = dot(vx, vw, bench=num_repeat)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([y], feed_dict = {vx: x.cpu().detach().numpy(), vw: w})
  sess.close()

def test(M, N, K, sparsity, mode, trans_a, trans_b, block):
  torch.manual_seed(1)
  AS0 = K if trans_a else M
  AS1 = M if trans_a else K
  BS0 = N if trans_b else K
  BS1 = K if trans_b else N
  shape = {'sdd': (M, N),
           'dsd': (AS0, AS1),
           'dds': (BS0, BS1)}[mode]
  # initialize mask
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  mask = generator.sample((shape[0]//block, shape[1]//block))
  # initialize inputs
  x = torch.rand((AS0, AS1), dtype=torch.float32, requires_grad=True).cuda()
  w = torch.rand((BS0, BS1), dtype=torch.float32, requires_grad=True).cuda()
  dy = torch.rand((M, N), dtype=torch.float32).cuda()
  x.retain_grad()
  w.retain_grad()
  # run
  ry, rdx, rdw = run_reference(x.clone(), w.clone(), mode, trans_a, trans_b, mask, block, dy)
  ty, tdx, tdw = run_triton(x.clone(), w.clone(), mode, trans_a, trans_b, mask, block, dy)
  # test
  assert(torch.allclose(ty, ry))
  assert(torch.allclose(tdx, rdx))
  assert(torch.allclose(tdw, rdw))
  # benchmark
  num_repeat = 100
  triton_ts = bench_triton(x.clone(), w.clone(), mode, trans_a, trans_b, mask, block, num_repeat)
  #openai_ts = bench_openai(x, bsz, mask, num_repeat)
  #flops = 2 * M * bsz * bsz * mask.nonzero().size(0)
  print(f'{rho*100}% sparse (block = {block}): {triton_ts*1e3:2.4f}ms')
  

# parameters
M, N, K = 256, 512, 384
block = 16
rhos = [0., 0.10, 0.25, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
#for rho in rhos:
#  test(M, N, K, rho, 'dds', False, False, block)

for mode in ['sdd', 'dsd', 'dds']:
  for trans_a in [False, True]:
    for trans_b in [False, True]:
      for rho in [0.5]:
        test(M, N, K, rho, mode, trans_a, trans_b, block)
          