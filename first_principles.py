import numpy as np
import random
from .engine import Value

# General Network Y = WX
'''
Bitnet
1. let Y = PX
2. P = P - avg(P)
3. let W = sign(P)
4. then, Y ~ BWX
'''
# Where B is a scaling factor that is computed in the paper
# but it can also be learned


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def parameters(self):
        return []

class ClassicNeuron(Module):
  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlin = nonlin

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    return act.tanh() if self.nonlin else act
  
  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class BitnetNeuron(Module):
    def __init__(self, nin, **kwargs):
        # Underlying weights are trainable Value objects
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    
    def __call__(self, x):
        # BitNet 1.58b weight quantization logic
        # 1. Mean Centering
        alpha = sum(self.w) / len(self.w)
        w_centered = [wi - alpha for wi in self.w]
        
        # 2. Scaling factor beta (using .data for absolute value as Value doesn't have __abs__)
        beta = sum(abs(wi.data) for wi in self.w) / len(self.w)
        
        # 3. Quantization (using .data for sign as it's non-differentiable)
        # In a more advanced engine, we would use Straight-Through Estimator (STE)
        w_bin = [beta * (1 if wi.data > 0 else -1) for wi in w_centered]
        
        # 4. Dot product with inputs
        return sum((wi * xi for wi, xi in zip(w_bin, x)), Value(0))

    def parameters(self):
        return self.w

    def __repr__(self):
        return f"BitnetNeuron({len(self.w)})"

        
