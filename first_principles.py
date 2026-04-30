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


class ClassicNeuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum(wi*xi for wi, xi in zip(self.w,x)) + self.b
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]


class BitnetNeuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1, 1) for _ in range(nin)]
    
    @staticmethod
    def sign(arr):
        return [1 if u > 0 else -1 for u in arr]
    
    def __call__(self, x):
        # mean center
        alpha = sum(self.w) / len(self.w)
        w_centered = [wi - alpha for wi in self.w]
        
        # sign
        s = BitnetNeuron.sign(w_centered)
        
        # scale
        beta = sum(abs(wi) for wi in self.w) / len(self.w)
        
        # approximate weights
        w_bin = [beta * si for si in s]
        
        # dot product
        return sum(wi * xi for wi, xi in zip(w_bin, x))

        