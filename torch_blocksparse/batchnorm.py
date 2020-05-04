import triton
import torch
import torch_blocksparse
import math

class _batchnorm(torch.autograd.Function):

  fwd_src = """
void batchnorm_ymv(TYPE *Y, float *M, float *V,
                  float *RM, float* RV,
                  TYPE *X, float *G, float *B,
                  int N, float mu, float eps) {
  // pointers
  int c = get_program_id(0);
  int rm[TM] = 0 ... TM;
  TYPE* px[TM] = X + rm + c*N;
  TYPE* py[TM] = Y + rm + c*N;

#ifdef TRAINING
  float run_mean = *(RM + c);
  float run_var  = *(RV + c);

  // compute mean
  float accm[TM] = 0;
  for(int i = 0; i < N; i = i + TM)
    accm = accm + *(px + i);
  float mean = (float)accm[+] / N;
  *(M + c) = mean;
  *(RM + c) = (1 - mu)*run_mean + mu*mean;

  // compute variance
  float accv[TM] = 0;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    x = x - mean;
    accv = accv + x*x;
  }
  float var = (float)accv[+] / N;
  *(V + c) = var;
  *(RV + c) = (1 - mu)*run_var + mu*var;
#else
  float mean = *(RM + c);
  float var  = *(RV + c);
#endif

  // Normalize batch
  float gamma = *(G + c);
  float beta = *(B + c);
  float rstdg = 1 / sqrtf(var + eps) * gamma;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float y[TM] = (x - mean)*rstdg + beta;
    *(py + i) = y;
  }
}
"""

  bwd_src = """
void batchnorm_dxdgdb(TYPE *DX, float *DG, float *DB,
                      TYPE *DY, TYPE *X, float *G,
                      float *M, float *V,
                      int N, float epsilon) {
  // pointers
  int c = get_program_id(0);
  int rx[TM] = 0 ... TM;
  int offset = c*N;
  TYPE* px[TM]  =  X + rx + c*N;
  TYPE* pdy[TM] = DY + rx + c*N;
  TYPE* pdx[TM] = DX + rx + c*N;

  // fetch statistics
  float gamma = *(G + c);
  float mean = *(M + c);
  float var = *(V + c);
  float rstd = 1 / sqrtf(var + epsilon);

  // compute dgamma and dbeta
  float  acc_dg[TM] = 0;
  float  acc_db[TM] = 0;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float dy[TM] = *(pdy + i);
    acc_dg += dy*(x - mean)*rstd;
    acc_db += dy;
  }
  float dg = acc_dg[+];
  float db = acc_db[+];
  *(DG + c) = dg;
  *(DB + c) = db;

  // compute dx
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float dy[TM] = *(pdy + i);
    float xhat[TM] = (x - mean) * rstd;
    float xtmp[TM] = (xhat * dg + db) / N;
    float dx[TM] = (dy - xtmp) * rstd * gamma;
    *(pdx + i) = dx;
  }
}
"""

  fwd_kernel = dict()
  bwd_kernel = dict()

  @staticmethod
  def forward(ctx, x, running_mean, running_var, gamma, beta, training, momentum, eps):
    N, C, H, W = x.shape
    # lazy compilation of kernel
    key = (training, x.dtype)
    if key not in _batchnorm.fwd_kernel:
      defines = {'TM': 256, 'TYPE': x.dtype}
      if training:
        defines['TRAINING'] = True
      _batchnorm.fwd_kernel[key] = triton.kernel(_batchnorm.fwd_src, defines = defines, num_warps=[2,4,8])
    kernel = _batchnorm.fwd_kernel[key]
    # allocate outputs
    y    = torch.empty_like(x)
    mean = torch.empty(C, dtype=torch.float32, device=x.device)
    var  = torch.empty(C, dtype=torch.float32, device=x.device)
    # execute kernels
    grid = lambda opt: [C]
    kernel(y, mean, var, running_mean, running_var, x, gamma, beta, H*W*N, momentum, eps, grid=grid)
    # save
    ctx.save_for_backward(x, gamma, beta, mean, var)
    ctx.eps = eps
    return y

  @staticmethod
  def backward(ctx, dy):
    # lazy compilation of kernel
    key = (dy.dtype, )
    if key not in _batchnorm.bwd_kernel:
      _batchnorm.bwd_kernel[key] = triton.kernel(_batchnorm.bwd_src, defines = {'TM': 256, 'TYPE': dy.dtype}, num_warps=[4])
    kernel = _batchnorm.bwd_kernel[key]
    # retrieve info
    x, gamma, beta, mean, var = ctx.saved_tensors
    eps = ctx.eps
    # allocate result
    dx = torch.empty_like(x)
    dgamma = torch.empty_like(gamma)
    dbeta = torch.empty_like(beta)
    # execute
    N, C, H, W = x.shape
    kernel(dx, dgamma, dbeta, dy, 
           x, gamma, mean, var, 
           H*W*N, eps,
           grid = lambda opt: [C])
    return dx, None, None, dgamma, dbeta, None, None, None

class _to_nchw(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
      return torch_blocksparse.Conv2d.chwn_to_nchw(x).clone()
    
    @staticmethod
    def backward(ctx, dy):
      return torch_blocksparse.Conv2d.nchw_to_chwn(dy)

class _to_chwn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
      return torch_blocksparse.Conv2d.nchw_to_chwn(x).clone()
    
    @staticmethod
    def backward(ctx, dy):
      return torch_blocksparse.Conv2d.chwn_to_nchw(dy)


class BatchNorm2d(torch.nn.modules.batchnorm.BatchNorm2d):
    
    _batchnorm = _batchnorm.apply

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.use_torch = False

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        strides = input.stride()
        # CHWN
        if strides[0] == 1:
          if self.use_torch:
            x = _to_nchw.apply(input)
            output = torch.nn.functional.batch_norm(
                                _to_nchw.apply(input), self.running_mean, self.running_var, self.weight, self.bias,
                                self.training or not self.track_running_stats,
                                exponential_average_factor, self.eps)
            output = _to_chwn.apply(output)
          else:
            output = BatchNorm2d._batchnorm(input, self.running_mean, self.running_var, self.weight, self.bias, 
                                          self.training or not self.track_running_stats, 
                                          exponential_average_factor, self.eps)
        #NCHW
        elif strides[3] == 1:
          output = torch.nn.functional.batch_norm(
                              input, self.running_mean, self.running_var, self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
          output = super(BatchNorm2d, self).forward(input)
          return torch_blocksparse.Conv2d.nchw_to_chwn(output)
        return output
