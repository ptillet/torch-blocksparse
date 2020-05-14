import triton
import torch
import math

src = '''
   __global__ void NAME (TYPE* X __readonly __noalias __aligned(16),
                         TYPE* Y __readonly __noalias __aligned(16),
                         int N, int C, int HW,
                         int stride_xn  __multipleof(M_STRIDE_XN), 
                         int stride_xc  __multipleof(M_STRIDE_XC), 
                         int stride_xhw __multipleof(M_STRIDE_XHW),
                         int stride_yn  __multipleof(M_STRIDE_YN),
                         int stride_yc  __multipleof(M_STRIDE_YC), 
                         int stride_yhw __multipleof(M_STRIDE_YHW)){
    // ranges
    int _rn   [TN ] = get_program_id(0)*TN  + 0 ... TN;
    int _rc   [TC ] = get_program_id(1)*TC  + 0 ... TC;
    int _rhw  [THW] = get_program_id(2)*THW + 0 ... THW;
    // broadcast
    int rn  [TN, TC, THW] = _rn[:, newaxis, newaxis];
    int rc  [TN, TC, THW] = _rc[newaxis, :, newaxis];
    int rhw [TN, TC, THW] = _rhw[newaxis, newaxis, :];
    // pointers to x
    TYPE *px  [TN, TC, THW] = X + rn*STRIDE_XN + rc*STRIDE_XC + rhw*STRIDE_XHW;
    TYPE *py  [TN, TC, THW] = Y + rn*STRIDE_YN + rc*STRIDE_YC + rhw*STRIDE_YHW;
    // check in bounds
    bool check[TN, TC, THW] = rn < N && rc < C && rhw < HW;
    *?(check)py = *?(check)px;
}
'''

class _permute(torch.autograd.Function):

    kernels = dict()

    @staticmethod
    def strides(N, C, H, W, order):
        return {
            'CHWN': [1, N*W*H, N*W, N],
            'NCHW': [W*H*C, W*H, W, 1]
        }[order]

    @staticmethod
    def multiple_of(N):
        if N % 8 == 0:
            return 8
        if N % 4 == 0:
            return 4
        if N % 2 == 0:
            return 2
        return 1
        
    @staticmethod
    def do_work(x, in_order, out_order):
        x_inner_mul = _permute.multiple_of(x.shape['NCHW'.index(in_order[-1])])
        y_inner_mul = _permute.multiple_of(x.shape['NCHW'.index(out_order[-1])])
        key = (x.dtype, in_order, out_order, x_inner_mul, y_inner_mul)
        if key not in _permute.kernels:
            TN  = [32] if in_order[-1] == 'N' or out_order[-1] == 'N' else 1
            TC  = [32] if in_order[-1] == 'C' or out_order[-1] == 'C' else 1
            THW = [32] if in_order[-1] == 'W' or out_order[-1] == 'W' else 1
            defines = {
                'NAME'        : f'permute_{in_order}_{out_order}_{x_inner_mul}_{y_inner_mul}',
                'TYPE'        : x.dtype,
                # stride multiple for X
                'M_STRIDE_XN' : 1 if in_order[-1]=='N' else x_inner_mul,
                'M_STRIDE_XC' : 1 if in_order[-1]=='N' else x_inner_mul,
                'M_STRIDE_XHW': 1 if in_order[-1]=='N' else x_inner_mul,
                # stride multiple for Y
                'M_STRIDE_YN' : 1 if out_order[-1]=='N' else y_inner_mul,
                'M_STRIDE_YC' : 1 if out_order[-1]=='N' else y_inner_mul,
                'M_STRIDE_YHW': 1 if out_order[-1]=='N' else y_inner_mul,
                # strides for X
                'STRIDE_XN'   : 1 if in_order[-1]=='N' else 'stride_xn',
                'STRIDE_XC'   : 1 if in_order[-1]=='C' else 'stride_xc',
                'STRIDE_XHW'  : 1 if in_order[-1]=='W' else 'stride_xhw',
                # strides for Y
                'STRIDE_YN'   : 1 if out_order[-1]=='N' else 'stride_yn',
                'STRIDE_YC'   : 1 if out_order[-1]=='C' else 'stride_yc',
                'STRIDE_YHW'  : 1 if out_order[-1]=='W' else 'stride_yhw',
                # tile parameters
                'TN'          : TN,
                'TC'          : TC,
                'THW'         : THW
            }
            _permute.kernels[key] = triton.kernel(src, defines=defines, num_warps=[4])
        kernel = _permute.kernels[key]
        N, C, H, W = x.shape
        y = torch.empty_strided(x.shape, _permute.strides(N, C, H, W, out_order), device=x.device, dtype=x.dtype)
        stride_xn, stride_xc, _, stride_xhw = x.stride()
        stride_yn, stride_yc, _, stride_yhw = y.stride()
        grid = lambda opt: (triton.cdiv(N, opt.d('TN')),
                            triton.cdiv(C, opt.d('TC')),
                            triton.cdiv(H*W, opt.d('THW')))
        kernel(x, y, N, C, H*W, stride_xn, stride_xc, stride_xhw, stride_yn, stride_yc, stride_yhw, grid=grid)
        return y

    @staticmethod
    def forward(ctx, x, in_order, out_order):
        y = _permute.do_work(x, in_order, out_order)
        ctx.in_order = in_order
        ctx.out_order = out_order
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dx = _permute.do_work(dy, ctx.out_order, ctx.in_order)
        return dx, None, None
        

class Permute(torch.nn.Module):

    def __init__(self, in_order, out_order):
        super(Permute, self).__init__()
        self.in_order = in_order
        self.out_order = out_order
    
    def forward(self, x):
        return _permute.apply(x, self.in_order, self.out_order)