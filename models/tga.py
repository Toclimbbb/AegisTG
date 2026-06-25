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
        # ==============================================================================
        # Step 1: Feature-Level Spatial Modulation (TGAttn)
        # ==============================================================================
        # Project visual map X and text descriptor G into low-rank subspaces (E and G')
        e = self.proj_w_v(x)
        g_prime = self.proj_w_t(g)
        
        # Calculate crossmodal affinity tensor (A_raw = E ⊗ G')
        a_raw = torch.matmul(e, g_prime.transpose(-1, -2))
        
        # Extract peak response along the text sequence length L (A_max)
        a_max, _ = torch.max(a_raw, dim=-1, keepdim=True)
        
        # Formulate non-linear spatial gating matrix (\hat{A})
        a_hat = self.gamma * torch.sigmoid((a_max / self.sqrt_c_h) + self.bias_b)
        
        # Depth-wise separable convolution (\Phi_{DS}) and element-wise modulation
        y_visual = self.phi_ds(x) * a_hat

        # ==============================================================================
        # Step 2: Object Query Generation
        # ==============================================================================
        # Decode candidate proposals (P) from the text-modulated visual features
        p = self.transformer_decoder(y_visual)

        # ==============================================================================
        # Step 3: Proposal-Level Grounding Alignment (ClassEmbed)
        # ==============================================================================
        # Apply L2 normalization to both features to eliminate amplitude interference
        p_norm = F.normalize(p, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        
        # Establish baseline logits with learnable temperature (\tau) and bias (\beta)
        logits_vision = torch.matmul(p_norm, t_norm.transpose(-1, -2)) * torch.exp(self.tau) + self.beta
        
        # Execute boundary filtering if validity Mask is provided
        if mask is not None:
            logits_vision = logits_vision.masked_fill(~mask, float('-inf'))
            
        # Eliminate false positives via secondary constraint (Global Cosine Similarity)
        # Achieves pristine manifold alignment closed loop
        global_cos_sim = torch.matmul(p_norm, t_norm.transpose(-1, -2))
        final_score = global_cos_sim * logits_vision
        
        return final_score, p
