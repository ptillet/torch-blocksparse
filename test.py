import blocksparse
import torch

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

torch.manual_seed(0)
# parameters
M, N, K = 256, 256, 256
BS = 16
# initialize inputs
mask = torch.randint(0, 2, (K//BS, N//BS))
x = torch.rand((M, K), dtype=torch.float32, requires_grad=True).cuda()
w = torch.rand((K, N), dtype=torch.float32, requires_grad=True).cuda()
x.retain_grad()
w.retain_grad()
# reference result
ry = reference_dot(x, w, mask)
dy = torch.rand_like(ry)
ry.backward(dy)
rdx = x.grad.clone()
rdw = w.grad.clone()
# reset gradients
x.grad.zero_()
w.grad.zero_()
# triton result
y_lut, y_locks, y_width = blocksparse._linear.make_ydx_lut(mask, BS)
dx_lut, dx_locks, dx_width = blocksparse._linear.make_ydx_lut(mask.T, BS)
dw_lut, dw_locks, dw_width = blocksparse._linear.make_dw_lut(mask, BS)
ty = blocksparse._linear.apply(x, w, BS, 
                               y_lut, y_locks, y_width,
                               dx_lut, dx_locks, dx_width,
                               dw_lut, dw_locks, dw_width)
ty.backward(dy)
tdx = x.grad.clone()
tdw = w.grad.clone()
x.grad.zero_()
w.grad.zero_()
# test
print((ty - ry).abs().max())
print((tdx - rdx).abs().max())
print((tdw - rdw).abs().max())


###############
# Test Module #
###############

linear = blocksparse.Linear(256, 256, BS, mask).cuda()
linear.weight.data.copy_(w)
nny = linear(x)
nny.backward(dy)
nndx = x.grad.clone()
nndw = linear.weight.grad.clone()
# test
print((nny - ry).abs().max())
print((nndx - rdx).abs().max())
print((nndw - rdw).abs().max())
