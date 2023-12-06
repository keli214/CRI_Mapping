import torch
from local_attention.local_attention import LocalAttention

if __name__ == "__main__":
  t, b, h, n, d = 4, 32, 4, 64, 384
  q = torch.rand(t, b, h, n, d//h)
  k = torch.rand(t, b, h, n, d//h)
  v = torch.rand(t, b, h, n, d//h)
  attn = LocalAttention(
          dim = d//h,
          window_size = d//h//3,       # window size. 512 is optimal, but 256 or 128 yields good enough results
          look_backward = 1,       # each window looks at the window before
          look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
          dropout = 0,           # post-attention dropout
          exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
        )
  mask = torch.ones(b, n).bool()
        
  x = attn(q,k,v, mask=mask)