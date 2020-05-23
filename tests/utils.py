import torch

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

def bench(fn, repeat):
  torch.cuda.synchronize()
  start = time()
  for i in range(repeat):
    fn()
  torch.cuda.synchronize()
  end = time()
  ret[3] = (end - start) / repeat

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