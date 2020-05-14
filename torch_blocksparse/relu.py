import triton
import torch
import torch_blocksparse
import math

class _relu(torch.autograd.Function):

  fwd_src = """
void relu_y(TYPE *X, TYPE *Y, TYPE scale, TYPE bias, TYPE* RES, int N) {
  int    off[TN] = get_program_id(0)*TN + 0 ... TN;
  // pointers
  TYPE*   px[TN] = X + off;
  TYPE*   py[TN] = Y + off;
  TYPE* pres[TN] = RES + off;
  // load
  bool check[TN] = off < N;
  TYPE     x[TN] = *?(check)px;
  TYPE   res[TN] = *?(check)pres;
  TYPE     y[TN] = x*scale + bias + res;
  // write-back
  *?(check)py    = (y > 0) ? y : 0;
}
"""

  bwd_src = """
void relu_dxdsdbdres(TYPE *X, TYPE *Y, TYPE scale,
                     TYPE *DX, TYPE *DY, TYPE* dscale, TYPE* dbias, TYPE* DRES, 
                     int N) {
  int    off[TN]  = get_program_id(0)*TN + 0 ... TN;
  // pointers
  TYPE*   pdx[TN] = DX + off;
  TYPE* pdres[TN] = DRES + off;
  TYPE*   pdy[TN] = DY + off;
  TYPE*    py[TN] = Y + off;
  TYPE*    px[TN] = X + off;
  // load
  bool check[TN]  = off < N;
  TYPE     y[TN]  = check ? *py : 0;
  TYPE    dy[TN]  = check ? *pdy : 0;
  TYPE     x[TN]  = check ? *px : 0;
  TYPE    du[TN]  = (y > 0) ? dy : 0;
  // write-back
  *?(check)pdx    = du * scale;
  *?(check)pdres  = du;
  f32_atomic_add(dbias, du[+]);
  f32_atomic_add(dscale, (du*x)[+]);
}
"""

  fwd_kernel = None
  bwd_kernel = None

  @staticmethod
  def forward(ctx, x, scale, bias, res):
    if _relu.fwd_kernel is None:
      defines = {'TYPE': x.dtype, 'TN': [256]}
      _relu.fwd_kernel = triton.kernel(_relu.fwd_src, defines=defines, num_warps=[4])
    kernel = _relu.fwd_kernel
    # launch kernel
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda opt: [triton.cdiv(N, opt.d('TN'))]
    kernel(x, y, scale.item(), bias.item(),res, N, grid=grid)
    # update context
    ctx.save_for_backward(x, y)
    ctx.scale = scale
    return y

  @staticmethod
  def backward(ctx, dy):
    # load from context
    x, y = ctx.saved_tensors
    # get kernel
    if _relu.bwd_kernel is None:
      defines = {'TYPE': x.dtype, 'TN': [256]}
      _relu.bwd_kernel = triton.kernel(_relu.bwd_src, defines=defines, num_warps=[4])
    kernel = _relu.bwd_kernel
    # allocate output
    dx = torch.empty_like(dy)
    dres = torch.empty_like(dy)
    dscale = torch.empty((1,), device=dy.device, dtype=dy.dtype)
    dbias = torch.empty_like(dscale)
    # launch kernel
    N = x.numel()
    grid = lambda opt: [triton.cdiv(N, opt.d('TN'))]
    kernel(x, y, ctx.scale.item(), dx, dy, dscale, dbias, dres, N, grid=grid)
    return dx, dscale, dbias, dres

relu = _relu.apply

class ReLU(torch.nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, scale, bias, residual):
        return relu(x, scale, bias, residual)
