from .linear import Linear
from .normalization import RMSNorm
from .first_principles import Module
from .engine import Value
import math

class BitnetSelfAttention(Module):
    def __init__(self, dim, n_heads, quant_type="bitnet158", bits=8):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # In BitNet, Q, K, V projections are quantized
        self.q_proj = Linear(dim, dim, quant_type=quant_type, bits=bits)
        self.k_proj = Linear(dim, dim, quant_type=quant_type, bits=bits)
        self.v_proj = Linear(dim, dim, quant_type=quant_type, bits=bits)
        self.o_proj = Linear(dim, dim, quant_type=quant_type, bits=bits)
        
    def __call__(self, x_seq):
        # x_seq: List[List[Value]] -> (seq_len, dim)
        seq_len = len(x_seq)
        
        # Projections
        q = [self.q_proj(xi) for xi in x_seq]
        k = [self.k_proj(xi) for xi in x_seq]
        v = [self.v_proj(xi) for xi in x_seq]
        
        # Simple Attention (Scalar implementation)
        # Note: This is O(seq_len^2 * dim) and very slow in a scalar engine
        out_seq = []
        scale = 1.0 / math.sqrt(self.head_dim)
        
        for i in range(seq_len):
            # Calculate attention scores for query i
            scores = []
            qi = q[i]
            for j in range(seq_len):
                kj = k[j]
                # Dot product qi * kj
                dot = sum((q_comp * k_comp for q_comp, k_comp in zip(qi, kj)), Value(0))
                scores.append((dot * scale).exp()) # Softmax numerator (approx)
            
            # Softmax denominator
            denom = sum(scores, Value(1e-10))
            probs = [s / denom for s in scores]
            
            # Weighted sum of values
            # result = sum(prob * v_j)
            context_vec = [Value(0) for _ in range(self.dim)]
            for j in range(seq_len):
                pj = probs[j]
                vj = v[j]
                context_vec = [cv + pj * v_comp for cv, v_comp in zip(context_vec, vj)]
            
            out_seq.append(self.o_proj(context_vec))
            
        return out_seq

    def parameters(self):
        return self.q_proj.parameters() + self.k_proj.parameters() + \
               self.v_proj.parameters() + self.o_proj.parameters()

class BitnetTransformerBlock(Module):
    def __init__(self, dim, n_heads, ff_dim, quant_type="bitnet158", bits=8):
        self.norm1 = RMSNorm(dim)
        self.attn = BitnetSelfAttention(dim, n_heads, quant_type=quant_type, bits=bits)
        self.norm2 = RMSNorm(dim)
        
        # Feed-forward network
        self.ff1 = Linear(dim, ff_dim, quant_type=quant_type, bits=bits)
        self.ff2 = Linear(ff_dim, dim, quant_type=quant_type, bits=bits)
        
    def __call__(self, x_seq):
        # x_seq: List[List[Value]]
        
        # 1. Attention + Residual
        norm_x = [self.norm1(xi) for xi in x_seq]
        attn_out = self.attn(norm_x)
        x_seq = [[xi_comp + ao_comp for xi_comp, ao_comp in zip(xi, ao)] 
                 for xi, ao in zip(x_seq, attn_out)]
        
        # 2. FFN + Residual
        norm_x2 = [self.norm2(xi) for xi in x_seq]
        ff_out = [self.ff2(self.ff1(xi)) for xi in norm_x2]
        x_seq = [[xi_comp + fo_comp for xi_comp, fo_comp in zip(xi, fo)] 
                 for xi, fo in zip(x_seq, ff_out)]
        
        return x_seq

    def parameters(self):
        return self.norm1.parameters() + self.attn.parameters() + \
               self.norm2.parameters() + self.ff1.parameters() + self.ff2.parameters()
