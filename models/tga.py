# ==============================================================================
# AegisTG: Text-Guided Alignment Network (TGA)
# [ANONYMOUS SUBMISSION FOR ICIG 2026 DOUBLE-BLIND REVIEW]
# ==============================================================================
# DECLARATION: 
# To comply with double-blind review policies and protect ongoing intellectual 
# property, this file exposes the core architectural forward-pass logic (Data Flow).
# The exact initialization parameters, kernel configurations, and spatial 
# evaluations within `__init__` are redacted. The complete, executable module 
# will be fully open-sourced upon official acceptance.
# ==============================================================================

def forward(self, x, g, t, mask=None):
        """
        Forward pass demonstrating the cascaded text-guided decoding and alignment pipeline:
        Feature-space modulation (TGAttn) and instance-level alignment (ClassEmbed).
        """
        e = self.proj_w_v(x)
        g_prime = self.proj_w_t(g)
        a_raw = torch.matmul(e, g_prime.transpose(-1, -2))
        a_max, _ = torch.max(a_raw, dim=-1, keepdim=True)
        a_hat = self.gamma * torch.sigmoid((a_max / self.sqrt_c_h) + self.bias_b)
        y_visual = self.phi_ds(x) * a_hat
        p = self.transformer_decoder(y_visual)
        p_norm = F.normalize(p, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        logits_vision = torch.matmul(p_norm, t_norm.transpose(-1, -2)) * torch.exp(self.tau) + self.beta
        if mask is not None:
            logits_vision = logits_vision.masked_fill(~mask, float('-inf'))
        global_cos_sim = torch.matmul(p_norm, t_norm.transpose(-1, -2))
        final_score = global_cos_sim * logits_vision
        
        return final_score, p
