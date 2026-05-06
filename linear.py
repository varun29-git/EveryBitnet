from .first_principles import Module, ClassicNeuron as Neuron
from .first_principles import BitnetNeuron as BN
from .bitlinear import BitnetNeuron158 as BN158

class Linear(Module):
    def __init__(self, nin, nout, quant_type="linear", **kwargs):
        self.quant_type = quant_type
        
        # Select neuron type based on quantization level
        if quant_type == "linear":
            neuron_cls = Neuron
        elif quant_type == "bitnet":
            neuron_cls = BN
        elif quant_type == "bitnet158":
            neuron_cls = BN158
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")
            
        self.neurons = [neuron_cls(nin, **kwargs) for _ in range(nout)] 
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Linear(nin={len(self.neurons[0].w)}, nout={len(self.neurons)}, type='{self.quant_type}')"
