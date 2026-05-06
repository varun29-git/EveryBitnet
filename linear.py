from .first_principles import Module, ClassicNeuron as Neuron
from .first_principles import BitnetNeuron as BN
from .bitnet_1_58b import BitnetNeuron158 as BN158
from .bitnet_v2 import BitnetNeuronV2 as BNV2

class Linear(Module):
    def __init__(self, nin, nout, quant_type="linear", bits=8, **kwargs):
        self.quant_type = quant_type
        self.bits = bits
        
        # Select neuron type based on quantization level
        if quant_type == "linear":
            neuron_cls = Neuron
        elif quant_type == "bitnet":
            neuron_cls = BN
        elif quant_type == "bitnet158":
            neuron_cls = BN158
        elif quant_type == "bitnet_v2":
            neuron_cls = BNV2
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")
            
        self.neurons = [neuron_cls(nin, bits=bits, **kwargs) for _ in range(nout)] 
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Linear(nin={len(self.neurons[0].w)}, nout={len(self.neurons)}, type='{self.quant_type}', bits={self.bits})"
