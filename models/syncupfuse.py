# ==============================================================================
# AegisTG: Synchronized Upsampling Fusion (SyncUpFuse)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class SyncUpFuse(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4, use_residual=True):
        super().__init__()
        
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if len(in_channels) < 2:
            raise ValueError("PolyFuse expects at least two input feature maps")

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.use_residual = use_residual

        self.align_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for c in self.in_channels
        )

        hidden = max(out_channels // reduction_ratio, 16)
        self.route_head = nn.Sequential(
            nn.Conv2d(out_channels * len(self.in_channels), hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(self.in_channels), 1, bias=True),
        )

        self.refine_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features):
        if not isinstance(features, (list, tuple)):
            raise TypeError("PolyFuse expects a list/tuple of feature maps as input")
        if len(features) != len(self.align_proj):
            raise ValueError(f"PolyFuse expected {len(self.align_proj)} inputs, got {len(features)}")

        target_size = features[0].shape[-2:]
        aligned_feats = []

        for proj, feat in zip(self.align_proj, features):
            feat = proj(feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            aligned_feats.append(feat)

        concat_feats = torch.cat(aligned_feats, dim=1)
        routing_weights = torch.softmax(self.route_head(concat_feats), dim=1)
        
        fused_feat = sum(
            feat * routing_weights[:, idx:idx + 1] 
            for idx, feat in enumerate(aligned_feats)
        )

        if self.use_residual:
            fused_feat = fused_feat + aligned_feats[0]

        return self.refine_out(fused_feat)
