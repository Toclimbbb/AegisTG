# ==============================================================================
# AegisTG: Reliability-Aware Asymmetric Fusion (RAAF)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradientExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        feat_avg = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(feat_avg, self.sobel_x, padding=1)
        grad_y = F.conv2d(feat_avg, self.sobel_y, padding=1)
        grad_mag = torch.abs(grad_x) + torch.abs(grad_y)
        return grad_mag


class GeometricBoundaryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_extractor = SobelGradientExtractor()
        self.conv_attn = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_d):
        pool_avg = torch.mean(feat_d, dim=1, keepdim=True)
        pool_max, _ = torch.max(feat_d, dim=1, keepdim=True)
        feat_edge = self.grad_extractor(feat_d)
        attn_cat = torch.cat([pool_avg, pool_max, feat_edge], dim=1)
        attn_map = self.conv_attn(attn_cat)
        return self.sigmoid(attn_map)


class CompetitiveWeightGenerator(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        hidden_channels = max(in_channels // reduction_ratio, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class RAAF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        hidden_dim = max(channels // 4, 16)
        self.reliability_branch = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.geom_attn_d = GeometricBoundaryAttention()
        self.spatial_attn_rgb = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), 
            nn.Sigmoid()
        )

        self.geom_injector = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.semantic_compensator = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.weight_gen_rgb = CompetitiveWeightGenerator(channels, reduction_ratio=4)
        self.weight_gen_d = CompetitiveWeightGenerator(channels, reduction_ratio=4)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, inputs):
        f_rgb, f_d = inputs[0], inputs[1]
        
        attn_d = self.geom_attn_d(f_d)   
        f_d = f_d * attn_d        
        
        pool_avg_rgb = torch.mean(f_rgb, dim=1, keepdim=True)
        pool_max_rgb, _ = torch.max(f_rgb, dim=1, keepdim=True)
        attn_rgb = self.spatial_attn_rgb(torch.cat([pool_avg_rgb, pool_max_rgb], dim=1))
        f_rgb = f_rgb * attn_rgb

        conf_d = self.reliability_branch(f_d)

        f_diff = f_d - f_rgb
        
        f_rgb_enh = f_rgb + conf_d * self.geom_injector(f_diff)
        f_d_enh = f_d + (1.0 - conf_d) * self.semantic_compensator(-f_diff)

        logits_rgb = self.weight_gen_rgb(f_rgb_enh)
        logits_d = self.weight_gen_d(f_d_enh)   
        
        stacked_logits = torch.stack([logits_rgb, logits_d], dim=1) 
        weights = F.softmax(stacked_logits, dim=1) 
        
        weight_rgb = weights[:, 0, :, :, :]
        weight_d = weights[:, 1, :, :, :]

        f_fuse = (f_rgb_enh * weight_rgb) + (f_d_enh * weight_d)
        f_out = self.output_conv(f_fuse)
        
        return f_out
