import random
import math
from .engine import Value
from .first_principles import Module

def fast_hadamard_transform(x):
    """
    Perform the Fast Hadamard Transform on a list of Values.
    The length of x must be a power of 2.
    """
    n = len(x)
    res = list(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x_val = res[j]
                y_val = res[j + h]
                res[j] = x_val + y_val
                res[j + h] = x_val - y_val
        h *= 2
    return res

class BitnetNeuronV2(Module):
    def __init__(self, nin, bits=4, **kwargs):
        self.nin = nin
        self.bits = bits
        # Hadamard requires power of 2 dimensions
        self.n_padded = 1 << (nin - 1).bit_length()
        
        # Latent weights for the padded dimension
        self.w = [Value(random.uniform(-1, 1)) for _ in range(self.n_padded)]
        self.scale_h = 1.0 / math.sqrt(self.n_padded)
        
    def __call__(self, x):
        # 1. Padding input to power of 2
        x_padded = x + [Value(0)] * (self.n_padded - len(x))
        
        # 2. Hadamard Transformation
        # Spreads activation energy across all dimensions to eliminate outliers
        x_had = fast_hadamard_transform(x_padded)
        x_had = [xi * self.scale_h for xi in x_had]
        
        # 3. Activation Quantization (Absmax to n-bits) with STE
        q_b = 2**(self.bits - 1) - 1
        gamma_x = max(abs(xi.data) for xi in x_had) + 1e-5
        x_quant = [(xi * (q_b / gamma_x)).ste_round() for xi in x_had]
        
        # 4. Ternary Weight Quantization (1.58-bit) with STE
        gamma_w = sum(abs(wi.data) for wi in self.w) / len(self.w) + 1e-5
        w_ternary = [gamma_w * (wi / gamma_w).ste_round() for wi in self.w]
        
        # 5. Dot product in the Hadamard-transformed space
        return sum((wi * xi for wi, xi in zip(w_ternary, x_quant)), Value(0))

    def parameters(self):
        return self.w

    def __repr__(self):
        return f"BitnetNeuronV2(nin={self.nin}, bits={self.bits})"
