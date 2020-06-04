import triton
import torch
import torch_blocksparse
import math

class _relu(torch.autograd.Function):

  fwd_src = """
void relu_y(TYPE *X __readonly  __noalias __aligned(16),  
            TYPE *Y __writeonly  __noalias __aligned(16), 
            TYPE scale, TYPE bias, 
            TYPE* RES __readonly  __noalias __aligned(16), int N) {
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
void relu_dxdsdbdres(TYPE *X __readonly  __noalias __aligned(16), 
                     TYPE *Y __readonly  __noalias __aligned(16), 
                     TYPE scale,
                     TYPE *DX __writeonly  __noalias __aligned(16), TYPE *DY, float* dscale, float* dbias, 
                     TYPE* DRES __writeonly __noalias __aligned(16), 
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
  f32_atomic_add(dbias, ((float[TN])du)[+]);
  f32_atomic_add(dscale, ((float[TN])(du*x))[+]);
}
"""

  fwd_kernel = dict()
  bwd_kernel = dict()

  @staticmethod
  def forward(ctx, x, scale, bias, res):
    if x.dtype not in _relu.fwd_kernel:
      defines = {'TYPE': x.dtype, 'TN': [128]}
      _relu.fwd_kernel[x.dtype] = triton.kernel(_relu.fwd_src, defines=defines, num_warps=[4])
    kernel = _relu.fwd_kernel[x.dtype]
    # launch kernel
    y = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
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
    if x.dtype not in _relu.bwd_kernel:
      defines = {'TYPE': x.dtype, 'TN': [128]}
      _relu.bwd_kernel[x.dtype] = triton.kernel(_relu.bwd_src, defines=defines, num_warps=[4])
    kernel = _relu.bwd_kernel[x.dtype]
    # allocate output
    dx = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    dres = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    dscale = torch.zeros((1,), device=dy.device, dtype=torch.float32)
    dbias = torch.zeros_like(dscale)
    # launch kernel
    N = x.numel()
    grid = lambda opt: [triton.cdiv(N, opt.d('TN'))]
    kernel(x, y, ctx.scale.item(), dx, dy, dscale, dbias, dres, N, grid=grid)
    return dx, dscale.type(x.dtype), dbias.type(x.dtype), dres

relu = _relu.apply

class ReLU(torch.nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, scale, bias, residual):
        return relu(x, scale, bias, residual)
