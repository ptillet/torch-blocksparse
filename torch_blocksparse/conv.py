import triton
import torch
from .matmul import _sparse_matmul
import math

src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                        TYPE* B __readonly  __noalias __aligned(16),
                        TYPE* C __noalias __aligned(16),
                        // shapes
                        int H, int W, int R, int S, int CC __multipleof(BLOCK),
                        int N __multipleof(32), int P, int Q, int K __multipleof(BLOCK),
                        int pad_h, int pad_w,
                        int stride_h, int stride_w,
                        // a strides
                        int stride_na __multipleof(BLOCK),
                        int stride_ca __multipleof(BLOCK),
                        int stride_ha __multipleof(BLOCK),
                        int stride_wa __multipleof(BLOCK),
                        // c strides
                        int stride_nc __multipleof(BLOCK),
                        int stride_kc __multipleof(BLOCK),
                        int stride_hc __multipleof(BLOCK),
                        int stride_wc __multipleof(BLOCK),
                        // lut and locks
                        int* lut, int* locks, int nlocks) {
     /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
#ifdef DW
    int* header = lut + pid0 * 4;
    int  off_ck = *(header + 0);
    int  off_cc = *(header + 1);
    int  off_cr = *(header + 2);
    int  off_cs = *(header + 3);
    int  L      = N*P*Q / TZ;
    int lockid  = select(TZ > 1, 1, 0);
    int maxid   = TZ;
    int offk    = pid1*L;
    // pointers to A
    int* p_delta      = lut + get_num_programs(0)*4;
    int  ra_nhw[TL]   = offk + 0 ... TL;
    int* pa_delta[TL] = p_delta + offk + 0 ... TL;
    int  ra_c[TM]     = off_cc  * TM   + 0 ... TM;
    int  ra_hw[TL]    =  ra_nhw / N;
    int  ra_n [TL]    =  ra_nhw % N;
    int  ra_h [TL]    = (ra_hw  / Q)*stride_h;
    int  ra_w [TL]    = (ra_hw  % Q)*stride_w;
    TYPE* pa[TM, TL]  = A + ra_c[:, newaxis]           * STRIDE_CA
                          + off_cr                     * STRIDE_HA
                          + off_cs                     * STRIDE_WA
                          + (ra_h[newaxis, :] - pad_h) * STRIDE_HA
                          + (ra_w[newaxis, :] - pad_w) * STRIDE_WA 
                          + ra_n[newaxis, :]           * STRIDE_NA;
    // pointers to B
    int   b_delta[TL] = offk + 0 ... TL;
    int   rb_k[TN]    = off_ck * TN + 0 ... TN;
    TYPE* pb[TL, TN]  = B + rb_k[newaxis, :] * STRIDE_KB
                          + b_delta[:, newaxis] * STRIDE_NPQB;
    // bounds for bound-checking
    int  h_lo = 0 + pad_h - off_cr;
    int  h_hi = H + pad_h - off_cr;
    int  w_lo = 0 + pad_w - off_cs;
    int  w_hi = W + pad_w - off_cs;
    // bounds-checking
    bool checkal[TL] = ra_h >= h_lo && ra_h < h_hi && 
                       ra_w >= w_lo && ra_w < w_hi;
    bool checkam[TM] = 1;
#else
    // load LUT header
    int *header = lut + pid0 * 4;
    int a_offset = *(header + 0);
    int b_offset = *(header + 1);
    int L        = *(header + 2);
    int column   = *(header + 3);
    int lockid = 0;
    int maxid = 1;
    // initialize a pointers
    int rc_npq[TM]    = (pid1 * TM) + 0 ... TM;
    int rc_pq [TM]    = rc_npq  / N;
    int rc_n  [TM]    = rc_npq  % N;
    int rc_p  [TM]    = rc_pq   / Q;
    int rc_q  [TM]    = rc_pq   % Q;
    int* pa_delta = lut + a_offset;
    int a_delta  __multipleof(TL) = *pa_delta;
    int ra_n  [TM]    = rc_n;
#ifdef DX
    int ra_h_0[TM]    = rc_p * stride_h + pad_h;
    int ra_w_0[TM]    = rc_q * stride_w + pad_w;
#else
    int ra_h_0[TM]    = rc_p * stride_h - pad_h;
    int ra_w_0[TM]    = rc_q * stride_w - pad_w;
#endif
    int ra_c  [TL]    = 0 ... TL;
    int offa[TM, TL]  = a_delta + ra_n  [:, newaxis] * STRIDE_NA
                                + ra_h_0[:, newaxis] * STRIDE_HA
                                + ra_w_0[:, newaxis] * STRIDE_WA
                                + ra_c  [newaxis, :] * STRIDE_CA;
    TYPE* pa[TM, TL]  = A + offa;
    // initialize b pointers
    int  rb_k[TN]     = 0 ... TN;
    int  rb_c[TL]     = 0 ... TL;
    int* pb_delta     = lut + b_offset;
    int  b_delta __multipleof(TL) = *pb_delta;
    TYPE* pb[TL, TN]  = B + b_delta + rb_k[newaxis, :] * STRIDE_BK
                                    + rb_c[:, newaxis] * STRIDE_BC;
    // prefetch
    int r = *(pa_delta + 1);
    int s = *(pa_delta + 2);
#ifdef DX
      int ra_h[TM] = ra_h_0 - r;
      int ra_w[TM] = ra_w_0 - s;
#else
      int ra_h[TM] = ra_h_0 + r;
      int ra_w[TM] = ra_w_0 + s;
#endif
    // bounds-checking
    bool checkam[TM]    = ra_h >= 0 && ra_h < H && 
                          ra_w >= 0 && ra_w < W;
    bool checkal[TL]    = 1;
#endif
    bool checka[TM, TL] = checkam[:, newaxis] && checkal[newaxis, :];
    bool checkb[TL, TN] = 1;
    TYPE a[TM, TL] = checka ? *pa : 0;
    TYPE b[TL, TN] = checkb ? *pb : 0;

    /* ---------------- */
    /*    Inner Loop    */
    /* ---------------- */
    // create result tile
    float acc[TM, TN] = 0;
    for(int l = L; l > 0; l -= TL) {
      acc += a @ b;
#ifdef DW
      int a_delta[TL] = *pa_delta;
      pa_delta += TL;
      pa += a_delta[newaxis, :];
      pb += TL * STRIDE_NPQB;
      ra_nhw += TL;
      ra_hw   =  ra_nhw / N;
      ra_h    = (ra_hw  / Q)*stride_h;
      ra_w    = (ra_hw  % Q)*stride_w;
      bool checkam[TM] = 1;
      bool checkal[TL] = ra_h >= h_lo && ra_h < h_hi && 
                         ra_w >= w_lo && ra_w < w_hi &&
                         (bool[TL])(l > TL);
#else
      // update pointers
      pa_delta += 3;
      pb_delta += 1;
      int a_delta __multipleof(TL) = *pa_delta;
      int b_delta __multipleof(TL) = *pb_delta;
      pa += a_delta;
      pb += b_delta;
      int r = *(pa_delta + 1);
      int s = *(pa_delta + 2);
#ifdef DX
      int ra_h[TM] = ra_h_0 - r;
      int ra_w[TM] = ra_w_0 - s;
#else
      int ra_h[TM] = ra_h_0 + r;
      int ra_w[TM] = ra_w_0 + s;
#endif
      bool checkam[TM] = ra_h >= 0 && ra_h < H &&
                         ra_w >= 0 && ra_w < W; 
      bool checkal[TL] = 1;
#endif
      // pre-fetch
      bool do_prefetch[TL] = l > TL;
      bool checka[TM, TL] = checkam[:, newaxis] && checkal[newaxis, :] && 
                            do_prefetch[newaxis, :];
      bool checkb[TL, TN] = do_prefetch[:, newaxis];
      a = checka ? *pa : 0;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize y pointers
#ifdef DW
    int  rc_c[TM] = 0 ... TM;
    int  rc_k[TN] = 0 ... TN;
    TYPE* pc[TM, TN] = C + pid0 * BLOCK * BLOCK
                         + rc_k[newaxis, :] * BLOCK
                         + rc_c[:, newaxis] * 1;
    bool checkc[TM, TN] = 1;
#else
    int rc_k[TN]     = column * TN + 0 ... TN;
    int offc[TM, TN] =     rc_n[:, newaxis] * STRIDE_NC
                         + rc_p[:, newaxis] * STRIDE_HC
                         + rc_q[:, newaxis] * STRIDE_WC
                         + rc_k[newaxis, :] * STRIDE_KC;
    TYPE* pc[TM, TN] = C + offc;
    bool checkc[TM, TN] = rc_npq[:, newaxis] < N*P*Q;
#endif
    // write-back directly
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    // accumulate partial result using spin-locks
    else {
      int *plock = locks + get_program_id(0)*nlocks + lockid - 1;
      int *pcount = plock + get_num_programs(0)*nlocks;
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % maxid);
      atomic_xchg(plock, 0);
    }
  }
'''

class _sparse_conv2d(torch.autograd.Function):

  _step = 16

  ##########################
  # UTILITIES              #
  ##########################
  locks = None
  @staticmethod
  def get_locks(size):
    if _sparse_conv2d.locks is None or size > _sparse_conv2d.locks.size(0):
      _sparse_conv2d.locks = torch.zeros(size, dtype=torch.int32).cuda()
    return _sparse_conv2d.locks

  @staticmethod
  def make_dds_lut(layout, block, step, is_dx, strides, full_layout, off_bh, off_bw, stride_bh, stride_bw):
    headers  = torch.empty((0,), dtype=torch.int64)
    a_deltas = torch.empty((0,), dtype=torch.int64)
    b_deltas = torch.empty((0,), dtype=torch.int64) 
    a_deltas_start = 0
    width = 0
    div = block // step
    # pointer increments for b
    if is_dx:
      size = layout.sum()
      # blocks are stored in order KRSC
      block_id = full_layout.clone().permute(0, 2, 3, 1).contiguous()
      block_id[block_id > 0] = 1 + torch.arange(full_layout.sum())
      # blocks are traversed in order CRSK
      block_id = block_id.permute(3, 1, 2, 0).contiguous()
      block_id = block_id[:, off_bh::stride_bh, off_bw::stride_bw, :]
      b_offset = block_id[block_id > 0] - 1
      b_offset = b_offset * block * block
      b_deltas = b_offset.clone()
      b_deltas[1:] -= b_offset[:-1]
      # starting position in delta table
      b_deltas_start = torch.empty((0,), dtype=torch.int64)
      current = torch.tensor([0])
      for i in range(layout.shape[1]):
        offset = layout[:, i, :, :].sum()
        if offset == 0:
          continue
        b_deltas_start = torch.cat((b_deltas_start, current))
        current += offset
      b_deltas[b_deltas_start] = b_offset[b_deltas_start]
    else:
      b_offset = torch.arange(layout.sum())
      b_offset = b_offset * block * block
      b_deltas = b_offset.clone()
      b_deltas[1:] -= b_offset[:-1]
      b_deltas = b_deltas.view(-1)
      b_deltas_start = torch.empty((0,), dtype=torch.int64)
      current = torch.tensor([0])
      for i in range(layout.shape[0]):
        offset = layout[i, :, :, :].sum()
        if offset == 0:
          continue
        b_deltas_start = torch.cat((b_deltas_start, current))
        current += offset
      b_deltas[b_deltas_start] = b_offset[b_deltas_start]
    # handle step
    b_deltas = b_deltas.view(-1, 1).repeat(1, div)
    if not is_dx:
      b_deltas[:, 1:] = step
      b_deltas[:, 0] -= (div-1)*step
    else:
      b_deltas[:, 1:] = step*block
      b_deltas[:, 0] -= (div - 1)*step*block
    b_deltas[b_deltas_start, 0] = b_offset[b_deltas_start]
    b_deltas = b_deltas.view(-1)
    b_deltas_start *= div
    # headers and pointer increments for a
    out_dim = 1 if is_dx else 0
    kk = 0
    for k in range(layout.shape[out_dim]):
        if is_dx:
          nnz = layout[:, k, :, :].permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*block*strides[0] - \
                      nnz[:,1]*strides[1] - \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[1:,2]*block*strides[0] - \
                      nnz[1:,1]*strides[1] - \
                      nnz[1:,0]*strides[2]
        else:
          nnz = layout[k, :, :, :].permute(1, 2, 0).nonzero()
          a_coffset = nnz[:,2]*block*strides[0] + \
                      nnz[:,1]*strides[1] + \
                      nnz[:,0]*strides[2]
          a_noffset = nnz[1:,2]*block*strides[0] + \
                      nnz[1:,1]*strides[1] + \
                      nnz[1:,0]*strides[2]
        if nnz.shape[0] == 0:
          continue
        a_inc  = a_noffset - a_coffset[:-1]
        a_inc  = torch.cat((a_coffset[:1], a_inc))
        # handle step
        offset = a_inc[0]
        a_inc = a_inc.view(-1, 1).repeat(1, div)
        a_inc[:, 1:] = step*strides[0]
        a_inc[:, 0] -= (div - 1)*step*strides[0]
        a_inc = a_inc.view(-1)
        a_inc[0] = offset
        # filter indices
        a_rr  = nnz[:, 0].view(-1, 1).repeat(1, div).view(-1)
        a_ss  = nnz[:, 1].view(-1, 1).repeat(1, div).view(-1)
        # build look-up table
        a_dd  = torch.stack((a_inc, a_rr, a_ss), dim=1).view(-1).contiguous()
        a_deltas = torch.cat((a_deltas, a_dd))
        # create headers
        size = nnz.shape[0]*div
        hh = torch.tensor([a_deltas_start, b_deltas_start[kk], size*step, k], dtype=torch.int64)
        a_deltas_start += 3*size
        headers = torch.cat((headers, hh))
        # update width
        width += 1
        kk += 1
    # create look-up table
    headers[0::4] += headers.shape[0]
    headers[1::4] += headers.shape[0] + a_deltas.shape[0]
    lut = torch.cat((headers, a_deltas, b_deltas)).type(torch.int32).contiguous().cuda()
    num_locks = 1
    return lut, num_locks, width
  
  @staticmethod
  def make_sdd_lut(layout, block):
    nnz = layout.permute(0, 2, 3, 1).contiguous().nonzero()
    width = layout.sum()
    # create lut
    k = nnz[:, 0]
    r = nnz[:, 1]
    s = nnz[:, 2]
    c = nnz[:, 3]
    lut = torch.stack((k, c, r, s), dim=1).view(-1).contiguous()
    lut = lut.type(torch.int32).cuda()
    # create locks
    num_locks = 1
    return lut, num_locks, width

  @staticmethod
  def unpack(idx, N, H, W, order):
    if order == 'CHWN':
      n  = idx % N
      hw = idx // N
      h  = hw // W
      w  = hw % W
    if order == 'NCHW':
      w  = idx % W
      nh = idx // W
      h  = nh % H
      n  = nh // H
    return n, h, w

  @staticmethod
  def make_db_delta(order, N, H, W, stride_n, stride_h, stride_w, step, 
                    transform_h = lambda h: h,
                    transform_w = lambda w: w):
    # aggregate reduction indices
    idx = torch.arange(N*H*W, dtype=torch.int32)
    next_idx = idx + step
    # unpacked reduction indices
    n, h, w = _sparse_conv2d.unpack(idx, N, H, W, order)
    next_n, next_h, next_w = _sparse_conv2d.unpack(next_idx, N, H, W, order)
    # transform indices
    h, next_h = transform_h(h), transform_h(next_h)
    w, next_w = transform_w(w), transform_w(next_w)
    # memory addresses
    off = w * stride_w + h * stride_h + n * stride_n
    next_off = next_w * stride_w + next_h * stride_h + next_n * stride_n
    # deltas
    ret = (next_off - off).cuda()
    return ret
    

  sdd_cache = dict()
  dds_cache = dict()
  @staticmethod
  def make_kernel(src, defines, cache, key, num_warps=[4]):
    if key not in cache:
      cache[key] = triton.kernel(src, defines=defines, num_warps=num_warps)
    return cache[key]

  ##########################
  # OPERATORS              #
  ##########################

  # Sparse = Dense x Dense
  @staticmethod
  def _sdd_conv2d(a, b, order, pad_h, pad_w, stride_h, stride_w,
                  num_blocks, layout, block, step, lut, num_locks, width, 
                  bench, time):
    # sanity checks
    a_dtype = a.dtype
    b_dtype = b.dtype
    Na, C, H, W = a.shape
    Nb, K, P, Q = b.shape
    _, _, R, S = layout.shape
    assert a_dtype == b_dtype
    assert Na == Nb
    c = torch.empty((num_blocks, block, block), dtype=a.dtype, device=a.device)
    # create kernel
    defines = {'NAME': 'sdd_conv2d', 
               'TYPE': a.dtype,
               'TM': block, 
               'TL': step, 
               'TN': block, 
               'BLOCK': block,
               'TZ': [1, 8, 16], 
               'DW': True,
               'STRIDE_NA': 1 if order[-1] == 'N' else 'stride_na',
               'STRIDE_CA': 1 if order[-1] == 'C' else 'stride_ca',
               'STRIDE_HA': 'stride_ha',
               'STRIDE_WA': 'stride_wa',
               'STRIDE_KB': 1 if order[-1] == 'C' else 'stride_kc',
               'STRIDE_NPQB': 1 if order[-1] == 'N' else 'K'}
    cache = _sparse_conv2d.sdd_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a_dtype), num_warps=[2, 4])
    # create semaphores
    locks = _sparse_conv2d.get_locks(2*width*num_locks)
    # create output
    stride_na, stride_ca, stride_ha, stride_wa = a.stride()
    stride_nc, stride_kc, stride_pc, stride_qc = b.stride()
    kernel(a, b, c, 
          H, W, R, S, C,
          Na, P, Q, K,
          pad_h, pad_w, stride_h, stride_w,
          stride_na, stride_ca, stride_ha, stride_wa,
          stride_nc, stride_kc, stride_pc, stride_qc,
          lut, locks, num_locks, 
          grid = lambda opt: [width, opt.d('TZ')], 
          bench = bench)
    return c

  @staticmethod
  def pad(tensor, pad):
      pad = pad + [0] *  (2*len(tensor.shape) - len(pad))
      begin = [ x if x > 0 else None for x in pad[-1::-2]]
      end   = [-x if x > 0 else None for x in pad[-2::-2]]
      slices = [slice(b, e) for b, e in zip(begin, end)]
      tensor = torch.nn.functional.pad(tensor, pad, 'constant', 0).to(memory_format=torch.channels_last)
      tensor = tensor[slices]
      return tensor

  # Dense = Dense x Sparse
  @staticmethod
  def _dds_conv2d(a, b, order, nchwkrspq,
                  pad_h, pad_w, stride_h, stride_w,
                  is_dx, layout, block, 
                  step, lut, num_locks, width, da_offs,
                  bench, time):
    N, C, H, W, K, R, S, P, Q = nchwkrspq
    # swap shapes
    if is_dx:
      C, K = K, C
      H, P = P, H
      W, Q = Q, W
    # create kernel
    defines = {'NAME': 'dds_conv2d_' + ('_dx' if is_dx else '_y'), 
               'TYPE': a.dtype,
               'TM': [128], 
               'TL': step, 
               'TN': block, 
               'BLOCK': block,
               'STRIDE_BK': 1 if is_dx else block,
               'STRIDE_BC': block if is_dx else 1,
               'STRIDE_NA': 1 if order[-1] == 'N' else 'stride_na',
               'STRIDE_CA': 1 if order[-1] == 'C' else 'stride_ca',
               'STRIDE_HA': 'stride_ha',
               'STRIDE_WA': 'stride_wa',
               'STRIDE_NC': 1 if order[-1] == 'N' else 'stride_nc',
               'STRIDE_KC': 1 if order[-1] == 'C' else 'stride_kc',
               'STRIDE_HC': 'stride_hc',
               'STRIDE_WC': 'stride_wc'}
    if is_dx:
      defines['DX'] = True
      defines['TRANSFORM_H'] = 'rc_p * stride_h + pad_h'
    cache = _sparse_conv2d.dds_cache
    kernel = _sparse_conv2d.make_kernel(src, defines, cache, (block, a.dtype, is_dx), num_warps=[2, 4])
    # create output
    if order == 'NHWC':
      c = torch.zeros(N, P, Q, K, dtype=a.dtype, device=a.device).permute(0,3,1,2)
    if order == 'CHWN':
      c = torch.zeros(K, P, Q, N, dtype=a.dtype, device=a.device).permute(3,0,1,2)
    stride_na, stride_ca, stride_ha, stride_wa = a.stride()
    if is_dx:
      for da_lut, da_num_locks, da_width, (a_pad_h, a_pad_w, off_bh, off_bw, off_ch, off_cw) in zip(lut, num_locks, width, da_offs):
        if da_lut is None:
          c[:, :, off_ch::stride_h, off_cw::stride_w] = 0
        else:
          da_locks = _sparse_conv2d.get_locks(2*da_width*da_num_locks*N*P*Q)
          cc = c[:, :, off_ch::stride_h, off_cw::stride_w]
          stride_nc, stride_kc, stride_pc, stride_qc = cc.stride()
          N, K, P, Q = cc.shape
          kernel(a, b, cc,
                H, W, R, S, C,
                N, P, Q, K,
                a_pad_h, a_pad_w, 
                1, 1,
                stride_na, stride_ca, stride_ha, stride_wa,
                stride_nc, stride_kc, stride_pc, stride_qc,
                da_lut, da_locks, da_num_locks, 
                grid = lambda opt: [da_width, triton.cdiv(N*P*Q, opt.d('TM'))], 
                bench = bench)
    else:
      stride_nc, stride_kc, stride_pc, stride_qc = c.stride()
      locks = _sparse_conv2d.get_locks(2*width*num_locks*N*P*Q)
      kernel(a, b, c, 
            H, W, R, S, C,
            N, P, Q, K,
            pad_h, pad_w, stride_h, stride_w,
            stride_na, stride_ca, stride_ha, stride_wa,
            stride_nc, stride_kc, stride_pc, stride_qc,
            lut, locks, num_locks, 
            grid = lambda opt: [width, triton.cdiv(N*P*Q, opt.d('TM'))], 
            bench = bench)

    return c

  
  @staticmethod
  def forward(ctx, a, b, 
              order, nchwkrspq, pad_h, pad_w, stride_h, stride_w, 
              num_blocks, layout, block,
              c_step, c_lut,  c_num_locks,  c_width,
              da_step, da_lut, da_num_locks, da_width, da_offs,
              db_step, db_lut, db_num_locks, db_width,
              bench, c_time, da_time, db_time):
    if order[-1] == 'N' and a.stride(0) != 1:
      raise ValueError(f'Input layout does not match {order}')
    if order[-1] == 'C' and a.stride(1) != 1:
      raise ValueError(f'Input layout does not match {order}')
    # N, C, H, W, K, R, S, P, Q = nchwkrspq
    # ctx.save_for_backward(a, b)
    # return torch.empty(N, K, P, Q, dtype=a.dtype, device=a.device).contiguous(memory_format=torch.channels_last)
    c = _sparse_conv2d._dds_conv2d(a, b, order,
                                   nchwkrspq, pad_h, pad_w, stride_h, stride_w,
                                   False, layout, block, 
                                   c_step, c_lut, c_num_locks, c_width, None,
                                   bench, c_time)
    # save for backward
    ctx.save_for_backward(a, b)
    # da parameters
    ctx.da_step = da_step
    ctx.da_lut = da_lut
    ctx.da_num_locks = da_num_locks
    ctx.da_width = da_width
    ctx.da_time = da_time
    # db parameters
    ctx.db_step = db_step
    ctx.db_lut = db_lut
    ctx.db_num_locks = db_num_locks
    ctx.db_width = db_width
    ctx.db_time = db_time
    # conv parameters
    ctx.order = order
    ctx.nchwkrspq = nchwkrspq
    ctx.bench = bench
    ctx.block = block
    ctx.layout = layout
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.da_offs = da_offs
    ctx.num_blocks = num_blocks
    return c
  
  @staticmethod
  def backward(ctx, dc):
    if ctx.order[-1] == 'N' and dc.stride(0) != 1:
      raise ValueError(f'Input layout does not match {ctx.order}')
    if ctx.order[-1] == 'C' and dc.stride(1) != 1:
      raise ValueError(f'Input layout does not match {ctx.order}')
    # a, b = ctx.saved_tensors
    # da = torch.empty_like(a)
    # db = torch.empty_like(b)
    # retrieve from context
    a, b         = ctx.saved_tensors
    # gradients w.r.t. a
    da = None
    if ctx.needs_input_grad[0]:
      da = _sparse_conv2d._dds_conv2d(dc, b, ctx.order, ctx.nchwkrspq, ctx.pad_h, ctx.pad_w, ctx.stride_h, ctx.stride_w,
                       True, ctx.layout, ctx.block, 
                       ctx.da_step, ctx.da_lut, ctx.da_num_locks, ctx.da_width, ctx.da_offs,
                       ctx.bench, ctx.da_time)
    # gradients w.r.t. b
    db = None
    if ctx.needs_input_grad[1]:
      db = _sparse_conv2d._sdd_conv2d(a, dc, ctx.order, ctx.pad_h, ctx.pad_w, ctx.stride_h, ctx.stride_w,
                                      ctx.num_blocks, ctx.layout, ctx.block,
                                      ctx.db_step, ctx.db_lut, ctx.db_num_locks, ctx.db_width,
                                      ctx.bench, ctx.db_time)
    return da, db, None, None, None,\
           None, None, None, None,\
           None, None, None, None,\
           None, None, None, None,\
           None, None, None, None,\
           None, None, None, None,\
           None, None, None


class Conv2d(torch.nn.Module):

  sparse_conv2d = _sparse_conv2d.apply

  @staticmethod
  def nchw_to_chwn(x):
    return x.permute(1,2,3,0).contiguous().permute(3,0,1,2)
  
  @staticmethod
  def chwn_to_nchw(x):
    return x.contiguous()

  def precompute_lut(self, order, nchwkrspq, dtype, block, stride_a, 
                     stride_h, stride_w, pad_h, pad_w):
    # heurstics for loop increments in kernels
    c_step = {(torch.float32, 16): 16,
              (torch.float32, 32): 16,
              (torch.float32, 64): 8,
              (torch.float16, 16): 16,
              (torch.float16, 32): 32,
              (torch.float16, 64): 16}[(dtype, block)]
    da_step = c_step
    db_step = {(torch.float32, 16): 16,
              (torch.float32, 32): 16,
              (torch.float32, 64): 16,
              (torch.float16, 16): 64,
              (torch.float16, 32): 64,
              (torch.float16, 64): 32}[(dtype, block)]
    # hash to avoid recompiling
    key = (order, nchwkrspq, stride_a, stride_h, stride_w, pad_h, pad_w)
    N, C, H, W, K, R, S, P, Q = nchwkrspq
    # unpack strides
    stride_na, stride_ca, stride_ha, stride_wa = stride_a
    if order == 'NHWC':
      stride_nc, stride_kc, stride_pc, stride_qc = K*Q*P, 1, K*Q, K
    if order == 'CHWN':
      stride_nc, stride_kc, stride_pc, stride_qc = 1, N*P*Q, N*Q, N
    # fill cache
    if key not in self.lut_cache:
      # Look-up tables for forward pass
      c_lut,  c_num_locks, c_width  = _sparse_conv2d.make_dds_lut(self.layout, self.block, c_step, 
                                                                  False, [stride_ca, stride_wa, stride_ha], None, None, None, None, None)
      # Look-up tables for data gradient
      # have to be careful here
      # the gradient of strided conv is a conv over a sparse image
      # which can be decomposed as a set of smaller convs
      da_lut, da_num_locks, da_width = [], [], []
      da_offs = []
      for off_ch in range(stride_h):
        for off_cw in range(stride_w):
          off_bh = (off_ch + pad_h) % stride_h
          off_bw = (off_cw + pad_w) % stride_w
          a_pad_h = int((pad_h + (stride_h - 1)*off_ch) / stride_h)
          a_pad_w = int((pad_w + (stride_w - 1)*off_cw) / stride_w)
          if off_bh >= R or off_bw >= S:
            lut, num_locks, width = None, None, None
          else:
            curr_layout = self.layout[:, :, off_bh::stride_h, off_bw::stride_w]
            lut, num_locks, width = _sparse_conv2d.make_dds_lut(curr_layout, self.block, da_step, 
                                                                True, [stride_kc, stride_qc, stride_pc], self.layout, off_bh, off_bw, stride_h, stride_w)
          da_lut.append(lut)
          da_num_locks.append(num_locks)
          da_width.append(width)
          da_offs.append((a_pad_h, a_pad_w, off_bh, off_bw, off_ch, off_cw))
      # look-up tables for weight gradients
      db_lut, db_num_locks, db_width = _sparse_conv2d.make_sdd_lut(self.layout, self.block)
      db_delta_a = _sparse_conv2d.make_db_delta(order, N, P, Q, stride_na, stride_ha, stride_wa, db_step,
                                                transform_h = lambda h: h*stride_h - pad_h,
                                                transform_w = lambda w: w*stride_w - pad_w)
      db_lut = torch.cat((db_lut, db_delta_a))
      # store results
      self.lut_cache[key] = (c_step, c_lut, c_num_locks, c_width, \
                             da_step, da_lut, da_num_locks, da_width, da_offs, \
                             db_step, db_lut, db_num_locks, db_width)
    return self.lut_cache[key]

  def __init__(self, in_channels, out_channels, kernel_size, layout, block, padding = (0,0), stride = (1,1), order='NHWC', bias = False):
    if order not in ['NHWC', 'CHWN']:
      raise ValueError('Only NHWC and CHWN orders are supported')
    super(Conv2d, self).__init__()
    assert bias == False
    self.lut_cache = dict()
    self.layout = layout
    self.block = block
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.weight = torch.nn.Parameter(torch.Tensor(layout.sum(), block, block), requires_grad=True)
    self.bias = None
    self.order = order
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

  # def extra_repr(self):
  #   s = ''
  #   return s.format(**self.__dict__)

  # for each row in rows, find its index in the 2D matrix X
  # e.g.,
  # X = [[2,1,0],
  #      [9,7,4],
  #      [5,8,3]]
  # y = [[2,1,0],
  #      [5,8,3]]
  # returns [0, 2]
  @staticmethod
  def row_idx(X, rows):
    if X.numel() == 0 or rows.numel() == 0:
      return torch.tensor([], dtype=X.dtype, device=X.device)
    delta = (X[None, :, :] - rows[:, None, :]) == 0
    idx = delta.all(2).any(0).nonzero().flatten()
    return idx

  @staticmethod
  def update_layout(layout_a, tensor_a, layout_b, init_val):
    _, block, block = tensor_a.shape
    nnz_a = layout_a.permute(0,2,3,1).contiguous().nonzero()
    nnz_b = layout_b.permute(0,2,3,1).contiguous().nonzero()
    nnz = list(set(map(tuple, nnz_a.tolist())) & set(map(tuple, nnz_b.tolist())))
    nnz = torch.tensor(nnz, dtype=layout_a.dtype, device=layout_a.device)
    idx_a = Conv2d.row_idx(nnz_a, nnz)
    idx_b = Conv2d.row_idx(nnz_b, nnz)
    tensor_b = torch.empty([layout_b.sum(), block, block], device=tensor_a.device, dtype=tensor_a.dtype)
    tensor_b[:] = init_val
    tensor_b[idx_b, :, :] = tensor_a[idx_a, :, :]
    #print(tensor_b[idx_b,:,:] - tensor_a[idx_a,:,:])
    return tensor_b

  def clear_cache(self):
    self.lut_cache = dict()

  def __call__(self, a):
    N, Ca, H, W = a.shape
    K, Cb, R, S = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
    P = (H + 2*self.padding[0] - R)//self.stride[0] + 1
    Q = (W + 2*self.padding[1] - S)//self.stride[1] + 1
    nchwkrspq = N, Ca, H, W, K, R, S, P, Q
    if Ca != Cb:
      raise ValueError('Incompatible channels in data and weights')
    # look-up tables
    c_step, c_lut, c_num_locks, c_width,\
    da_step, da_lut, da_num_locks, da_width, da_offs,\
    db_step, db_lut, db_num_locks, db_width = self.precompute_lut(self.order, nchwkrspq, a.dtype, self.block, 
                                                         a.stride(), self.stride[0], self.stride[1],\
                                                         self.padding[0], self.padding[1])
    # run kernel
    c = Conv2d.sparse_conv2d(a, self.weight, self.order, nchwkrspq, 
                             self.padding[0], self.padding[1], self.stride[0], self.stride[1],
                             self.layout.sum(), self.layout.data, self.block,
                             c_step, c_lut, c_num_locks, c_width,
                             da_step, da_lut, da_num_locks, da_width, da_offs,
                             db_step, db_lut, db_num_locks, db_width,
                             False, [None], [None], [None])
    return c