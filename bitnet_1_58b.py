import random
from .engine import Value
from .first_principles import Module

class BitnetNeuron158(Module):
    def __init__(self, nin, **kwargs):
        # Latent weights are trainable Value objects
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        
    def __call__(self, x):
        # 1. Activation Quantization (Absmax to 8-bit)
        gamma_x = max(abs(xi.data) for xi in x) + 1e-5
        x_quant = [xi * (127.0 / gamma_x) for xi in x]
        
        # 2. Ternary Weight Quantization (BitNet 1.58b)
        # gamma_w is the mean of absolute weight values
        gamma_w = sum(abs(wi.data) for wi in self.w) / len(self.w) + 1e-5
        
        # Quantize weights to {-1, 0, 1}
        # Formula: Round(Clip(w / gamma, -1, 1))
        def ternary_quant(v, g):
            scaled = v.data / g
            return round(max(-1, min(1, scaled)))
            
        # These are the actual values used in the computation
        w_ternary = [gamma_w * ternary_quant(wi, gamma_w) for wi in self.w]
        
        # 3. Dot product with quantized activations and ternary weights
        return sum((wi * xi for wi, xi in zip(w_ternary, x_quant)), Value(0))

    def parameters(self):
        return self.w

    def __repr__(self):
        return f"BitnetNeuron158({len(self.w)})"
