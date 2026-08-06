# ==============================================================================
# AegisTG: Omnidimensional Feature Refinement (OmniMixer)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv


class WaveCore(nn.Module):
    def __init__(self, channels, ratio=4, min_channels=16, max_fft_size=16):
        super().__init__()
        hidden = max(channels // ratio, min_channels)
        hidden = min(hidden, channels)
        
        self.hidden_channels = hidden
        self.max_fft_size = max_fft_size

        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        
        self.mag_mod = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=True),
        )
        
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _get_fft_size(self, h, w):
        if self.max_fft_size is None or self.max_fft_size <= 0:
            return h, w
        scale = min(1.0, float(self.max_fft_size) / float(max(h, w)))
        return max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    def forward(self, x):
        _, _, h, w = x.shape
        
        x_sub = self.squeeze(x)
        
        fft_h, fft_w = self._get_fft_size(h, w)
        if (fft_h, fft_w) != (h, w):
            x_fft_in = F.adaptive_avg_pool2d(x_sub, (fft_h, fft_w))
        else:
            x_fft_in = x_sub

        dtype = x_fft_in.dtype
        x_freq = torch.fft.rfft2(x_fft_in.float(), norm="ortho")
        mag = torch.abs(x_freq)
        phase = x_freq / (mag + 1e-6)

        delta = self.mag_mod(torch.log1p(mag))
        mag = mag * (1.0 + torch.tanh(delta))

        x_out = torch.fft.irfft2(phase * mag, s=(fft_h, fft_w), norm="ortho")
        x_out = x_out.to(dtype=dtype)

        if (fft_h, fft_w) != (h, w):
            x_out = F.interpolate(x_out, size=(h, w), mode="bilinear", align_corners=False)

        return self.expand(x_out)


class OmniMixer(nn.Module):
    def __init__(
        self,
        inc,
        ouc,
        ratio=4,
        min_channels=16,
        max_fft_size=16,
        dilation=1,
        use_freq=True,
    ):
        super().__init__()
        hidden = max(inc // ratio, min_channels)
        hidden = min(hidden, inc)
        self.use_freq = use_freq

        self.norm_freq = nn.GroupNorm(1, inc)
        self.wave_core = WaveCore(
            inc, ratio=ratio, min_channels=min_channels, max_fft_size=max_fft_size
        ) if use_freq else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, inc, 1, 1))

        self.norm_spa = nn.GroupNorm(1, inc)
        self.spa_in = nn.Conv2d(inc, hidden * 2, 1, bias=False)
        
        self.spa_dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.spa_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spa_out = nn.Sequential(
            nn.Conv2d(hidden, inc, 1, bias=False),
            nn.BatchNorm2d(inc),
        )
        self.beta = nn.Parameter(torch.zeros(1, inc, 1, 1))

        self.conv_out = Conv(inc, ouc, k=1) if inc != ouc else nn.Identity()

    def forward(self, x):
        if self.use_freq:
            x = x + self.gamma * self.wave_core(self.norm_freq(x))

        y = self.norm_spa(x)
        y_val, y_gate = self.spa_in(y).chunk(2, dim=1)
        y = y_val * torch.sigmoid(y_gate)
        
        y = self.spa_dw(y)
        y = y * self.spa_attn(y)
        y = self.spa_out(y)

        return self.conv_out(x + self.beta * y)
