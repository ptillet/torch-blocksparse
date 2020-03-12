import torch_blocksparse
import torch
torch.manual_seed(0)

def reference_dot(x, w, mask):
  WS0, WS1 = w.size()
  MS0, MS1 = mask.size()
  assert WS0 % MS0 == 0
  assert WS1 % MS1 == 0
  block_size_0 = WS0 // MS0
  block_size_1 = WS1 // MS1
  assert block_size_0 == block_size_1
  maskedw = w.clone()
  for bi, wi in enumerate(range(0, WS0, block_size_0)):
    for bj, wj in enumerate(range(0, WS1, block_size_1)):
      maskedw[wi : wi+block_size_0,
              wj : wj+block_size_1] *= mask[bi, bj]
  return torch.matmul(x, maskedw)

def run_reference(x, w, mask, bsz, dy):
  y = reference_dot(x, w, mask)
  y.backward(dy)
  dx = x.grad.clone()
  dw = w.grad.clone()
  x.grad.zero_()
  w.grad.zero_()
  return y, dx, convert_weights(dw, mask, bsz)

def convert_weights(w, mask, bsz):
  # allocate output
  ret = torch.empty((mask.sum(), bsz, bsz), dtype=w.dtype, device=w.device)
  # fill output
  nnz = mask.nonzero()
  i, j = nnz[:, 0], nnz[:, 1]
  for idx, (ii, jj) in enumerate(zip(i, j)):
    ret[idx, :, :] = w[ii*bsz: (ii+1)*bsz, 
                       jj*bsz: (jj+1)*bsz]
  return ret

def run_triton(x, w, mask, bsz, dy):
  linear = torch_blocksparse.Linear(w.size(0), w.size(1), bsz, mask).cuda()
  linear.weight.data.copy_(convert_weights(w, mask, bsz))
  y = linear(x)
  y.backward(dy)
  dx = x.grad.clone()
  dw = linear.weight.grad.clone()
  x.grad.zero_()
  return y, dx, dw

def bench_triton(x, bsz, mask, num_repeat):
  from time import time
  linear = torch_blocksparse.Linear(w.size(0), w.size(1), bsz, mask).cuda()
  linear.bench_y = num_repeat
  linear.bench_dx = num_repeat
  linear.bench_dw = num_repeat
  y = linear(x)
  torch.cuda.synchronize()
  y = linear(x)
  torch.cuda.synchronize()
  return linear.timings.ty*1e-9
  
def bench_openai(x, bsz, mask, num_repeat):
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
  dot = BlocksparseMatMul(sparsity, block_size=bsz)
  # shapes
  M, K = x.size()
  N = mask.size(1)*bsz
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



# parameters
M, N, K = 1024, 1024, 1024
bsz = 16
rhos = [0., 0.10, 0.25, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
#rhos = [0.]
for sparsity in rhos:
    torch.manual_seed(1)
    # initialize mask
    probs = torch.Tensor([sparsity, 1-sparsity])
    generator = torch.distributions.categorical.Categorical(probs)
    mask = generator.sample((K//bsz, N//bsz))
    # initialize inputs
    x = torch.rand((M, K), dtype=torch.float32, requires_grad=True).cuda()
    w = torch.rand((K, N), dtype=torch.float32, requires_grad=True).cuda()
    dy = torch.rand((M, N), dtype=torch.float32).cuda()
    x.retain_grad()
    w.retain_grad()
    # run
    ry, rdx, rdw = run_reference(x, w, mask, bsz, dy)
    ty, tdx, tdw = run_triton(x, w, mask, bsz, dy)
    # test
    assert(torch.allclose(ty, ry))
    assert(torch.allclose(tdx, rdx))
    assert(torch.allclose(tdw, rdw))
    # benchmark
    num_repeat = 100
    triton_ts = bench_triton(x, bsz, mask, num_repeat)
    #openai_ts = bench_openai(x, bsz, mask, num_repeat)
    #flops = 2 * M * bsz * bsz * mask.nonzero().size(0)
    print(f'{sparsity*100}% sparsity (block = {bsz}): {triton_ts*1e3:2.4f}ms')