# ==============================================================================
# AegisTG: Reliability-Aware Asymmetric Fusion (RAAF)
# [ANONYMOUS SUBMISSION FOR ICIG 2026 DOUBLE-BLIND REVIEW]
# ==============================================================================
# DECLARATION: 
# To comply with double-blind review policies and protect ongoing intellectual 
# property, this file exposes the core architectural forward-pass logic (Data Flow).
# The exact initialization parameters, kernel configurations, and spatial 
# evaluations within `__init__` are redacted. The complete, executable module 
# will be fully open-sourced upon official acceptance.
# ==============================================================================
    def forward(self, inputs):
        """
        Forward pass demonstrating the architectural data flow:
        Evidence-driven differential compensation and competitive fusion.
        """
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
