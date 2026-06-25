# ==============================================================================
# AegisTG: Omnidimensional Feature Refinement (OmniMixer)
# [ANONYMOUS SUBMISSION FOR ICIG 2026 DOUBLE-BLIND REVIEW]
# ==============================================================================
# DECLARATION: 
# To comply with double-blind review policies and protect ongoing intellectual 
# property, this file exposes the core architectural forward-pass logic (Data Flow).
# The exact initialization parameters, kernel configurations, and low-rank FFT 
# operations within `__init__` are redacted. The complete, executable module 
# will be fully open-sourced upon official acceptance.
# ==============================================================================
    def forward(self, x):
        """
        Forward pass demonstrating the architectural data flow:
        Spatial-Frequency Dual-Domain Synergy.
        """
        # Step 1: Global context mixing via WaveCore (Frequency Domain)
        # Non-linear amplitude modulation to reconstruct high-frequency step edges.
        if self.use_freq:
            x = x + self.gamma * self.wave_core(self.norm_freq(x))

        # Step 2: Local context mixing (Spatial Domain)
        y = self.norm_spa(x)
        
        # Feature filtering mechanism via GLU (Gated Linear Unit) equivalent
        y_val, y_gate = self.spa_in(y).chunk(2, dim=1)
        y = y_val * torch.sigmoid(y_gate)
        
        # Spatial depth-wise convolution and channel attention enhancement
        y = self.spa_dw(y)
        y = y * self.spa_attn(y)
        y = self.spa_out(y)

        # Step 3: Final aggregation and channel alignment
        return self.conv_out(x + self.beta * y)
