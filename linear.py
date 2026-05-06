from .first_principles import Module, ClassicNeuron as Neuron
from .first_principles import BitnetNeuron as BN

class Linear(Module):
    def __init__(self, nin, nout, quant_type, **kwargs):
        neuron_cls = Neuron if (quant_type == "linear") else BN
        self.neurons = [neuron_cls(nin, **kwargs) for _ in range(nout)] 
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"LinearLayer({len(self.neurons)})"
