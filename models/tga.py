# ==============================================================================
# AegisTG: Text-Guided Alignment Network (TGA)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))

class TGAttn(nn.Module):
    def __init__(self, v_in, t_in, out_c, num_heads=1):
        super().__init__()
        self.nh = num_heads
        self.hd = out_c // num_heads
        self.sqrt_c_h = self.hd ** 0.5
        self.proj_w_v = nn.Conv2d(v_in, out_c, 1, bias=False) if v_in != out_c else nn.Identity()
        self.proj_w_t = nn.Linear(t_in, out_c, bias=True)
        self.bias_b = nn.Parameter(torch.zeros(num_heads))
        self.gamma = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.phi_ds = DepthwiseSeparableConv(v_in, out_c, 3)

    def forward(self, x, g):
        bs, _, h, w = x.shape
        e = self.proj_w_v(x)
        g_prime = self.proj_w_t(g)
        if g_prime.ndim == 2:
            g_prime = g_prime.unsqueeze(0).expand(bs, -1, -1)
        
        e = e.view(bs, self.nh, self.hd, h, w).permute(0, 1, 3, 4, 2).contiguous()
        g_prime = g_prime.view(bs, -1, self.nh, self.hd)
        
        a_raw = torch.einsum("bnhwc,blnc->bnhwl", e, g_prime)
        a_max = a_raw.max(dim=-1).values
        a_hat = self.gamma * torch.sigmoid((a_max / self.sqrt_c_h) + self.bias_b.view(1, -1, 1, 1))
        
        a_hat_exp = a_hat.repeat_interleave(self.hd, dim=1)
        return self.phi_ds(x) * a_hat_exp

class ClassEmbed(nn.Module):
    def __init__(self, init_bias=100.0, init_scale=15.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_scale).log())
        self.beta = nn.Parameter(torch.full((), -torch.log(torch.tensor(init_bias))))

    def forward(self, p, t, mask=None):
        if p.ndim == 4:
            p = p.flatten(2).transpose(1, 2)
        if t.ndim == 2:
            t = t.unsqueeze(0).expand(p.shape[0], -1, -1)

        p_norm = F.normalize(p, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        
        cos_sim = torch.matmul(p_norm, t_norm.transpose(-1, -2))
        logits = cos_sim * torch.exp(self.tau) + self.beta
        
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
            
        return cos_sim * logits

class CascadedTGA(nn.Module):
    def __init__(self, v_in=128, t_in=512, out_c=128, num_heads=1, tf_dec=None):
        super().__init__()
        self.tg_attn = TGAttn(v_in, t_in, out_c, num_heads)
        self.tf_dec = tf_dec if tf_dec is not None else nn.Identity()
        self.class_embed = ClassEmbed()

    def forward(self, x, g, t, mask=None):
        y_visual = self.tg_attn(x, g)
        p = self.tf_dec(y_visual)
        return self.class_embed(p, t, mask), p
