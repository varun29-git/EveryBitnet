import math
import random
from .engine import Value
from .first_principles import Module

class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        # Learnable scaling parameter (gamma)
        self.weight = [Value(1.0) for _ in range(dim)]

    def __call__(self, x):
        # x is a list of Value objects
        # 1. Calculate RMS: sqrt(mean(x^2) + eps)
        ssq = sum((xi**2 for xi in x), Value(0))
        rms = (ssq / len(x) + self.eps)**0.5
        
        # 2. Normalize and scale
        # y = (x / rms) * weight
        return [(xi / rms) * wi for xi, wi in zip(x, self.weight)]

    def parameters(self):
        return self.weight

    def __repr__(self):
        return f"RMSNorm({len(self.weight)})"
