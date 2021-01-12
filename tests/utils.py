import torch
from time import time

# convert dense matrix with explicit zeros to sparse matrix
def dense_to_sparse(w, mask, block):
  Z = w.size(0)
  ret = torch.empty((Z, mask.sum(), block, block), dtype=w.dtype, device=w.device)
  nnz = mask.nonzero(as_tuple=False)
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

def relerr(x, y, eps=1e-7):
  x = x.data.clone()
  y = y.data.clone()
  if x.shape != y.shape:
    return 1
  diff  = x - y + eps
  ewmax = torch.max(x.abs(), y.abs())
  return (diff.abs() / (ewmax + eps)).max().item()

def mempad(x, shape, strides, pad_size=1024*1024):
  pad = float('nan') * torch.ones(pad_size, device=x.device, dtype=x.dtype)
  chunk = torch.cat((pad, x.flatten(), pad))
  ret = chunk[pad_size:-pad_size].as_strided(shape, strides)
  return ret

def mask_weights(w, layout, block):
  repeat_k = block*torch.ones(layout.shape[0], dtype=torch.int64)
  repeat_c = block*torch.ones(layout.shape[1], dtype=torch.int64)
  mask = layout.repeat_interleave(repeat_k, dim=0)\
               .repeat_interleave(repeat_c, dim=1).cuda().type(w.dtype)
  return w * mask

def bench(fn, repeat, hook = None):
  torch.cuda.synchronize()
  # estimate hook time
  hook_time = 0
  if hook is not None:
    start = time()
    for i in range(repeat):
      hook()
    torch.cuda.synchronize()
    end = time()
    hook_time = end - start
  # run bench
  fn()
  torch.cuda.synchronize()
  start = time()
  for i in range(repeat):
    if hook is not None:
      hook()
    fn()
  torch.cuda.synchronize()
  end = time()
  return (end - start - hook_time) / repeat

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

def allclose(x, y):
  assert x.dtype == y.dtype
  rtol, atol = {torch.float32: (1e-4, 1e-2),
                torch.float16: (1e-2, 1e-3)}[x.dtype]
  return torch.allclose(x, y, rtol=rtol, atol=atol)

def make_layout(rho, shape):
  probs = torch.Tensor([rho, 1-rho])
  generator = torch.distributions.categorical.Categorical(probs)
  layout = generator.sample(shape)
  return layout


def nbytes(x):
  return x.nelement() * x.element_size()

def prettyprint(x, y, L, x_name = ' '):
  L = [x_name] + list(map(str, L))
  pad = max([len(x) for x in L]) + 2
  frmt = (f'{{:>{pad}}}')*len(L)
  print(frmt.format(*L))
  for i in range(y.shape[0]):
    row = [x[i]] + y[i,:].tolist()
    frmt = f'{{:>{pad}}}' + f'{{:{pad}.2f}}'*(len(L)-1)
    print(frmt.format(*row))