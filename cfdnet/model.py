class DecompositionBlock(nn.Module):
    """Decomposition block with multi-head attention, GLU gating, and residual connections."""
    def __init__(self, d_model, num_heads=8, num_functions=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_functions = num_functions
        
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
        self.psi_functions = nn.ModuleList([AdaptiveUnivariateFunction() for _ in range(num_functions)])
        
        # Changed the input features to num_functions to match the shape of transformed
        self.gate = nn.Sequential(
            nn.Linear(num_functions, d_model * 2),  
            nn.GLU(),
            nn.LayerNorm(d_model)
        )
        
        self.input_projection = nn.Linear(d_model, num_functions)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        
        projected = self.input_projection(x)
        
        transformed = torch.stack([f(projected[..., i]) for i, f in enumerate(self.psi_functions)], dim=-1)
        
        gated_output = self.gate(transformed)
        
        return x + attn_output + self.output_projection(gated_output)