# BitNet From First Principles

A small educational Python implementation of BitNet-style quantized neural network pieces built on top of a scalar automatic differentiation engine.

The code is intentionally simple and readable: tensors are represented as Python lists of `Value` objects, and operations are written in a first-principles style rather than using a deep learning framework.

## What's Included

- `engine.py` - a scalar autograd `Value` class with backpropagation support.
- `first_principles.py` - base `Module`, classic neurons, and a BitNet-style binary neuron.
- `bitnet_1_58b.py` - a ternary BitNet 1.58-bit neuron using a straight-through estimator.
- `bitnet_v2.py` - a BitNet v2-style neuron with Hadamard activation spreading and configurable activation bits.
- `linear.py` - a `Linear` layer that can switch between classic and BitNet quantized neurons.
- `normalization.py` - RMSNorm implemented with scalar `Value` objects.
- `transformer.py` - simple BitNet self-attention and transformer block components.
- `optimizer.py` - first-principles Adam and simplified Muon optimizers.

## Requirements

- Python 3.10+
- `numpy`

Install the dependency with:

```bash
pip install numpy
```

## Usage

Because the modules use relative imports, import them as a package from the parent directory of this repo:

```bash
cd ..
python
```

```python
from bitnet.engine import Value
from bitnet.linear import Linear
from bitnet.optimizer import Adam

model = Linear(2, 1, quant_type="bitnet158")

x = [Value(0.5), Value(-1.0)]
y = model(x)

target = Value(1.0)
loss = (y - target) ** 2

loss.backward()

opt = Adam(model.parameters(), lr=1e-2)
opt.step()
opt.zero_grad()

print(loss.data)
```

## Quantization Modes

`Linear` supports these `quant_type` values:

- `"linear"` - standard tanh neuron.
- `"bitnet"` - binary BitNet-style weights with absmax activation scaling.
- `"bitnet158"` - ternary `{-1, 0, 1}` weight quantization.
- `"bitnet_v2"` - Hadamard-transformed activations with configurable activation precision.

Example:

```python
layer = Linear(4, 3, quant_type="bitnet_v2", bits=4)
```

## Transformer Block

The transformer components operate on sequences of vectors represented as nested lists of `Value` objects:

```python
from bitnet.engine import Value
from bitnet.transformer import BitnetTransformerBlock

block = BitnetTransformerBlock(dim=4, n_heads=2, ff_dim=8, quant_type="bitnet158")

x_seq = [
    [Value(0.1), Value(0.2), Value(0.3), Value(0.4)],
    [Value(0.4), Value(0.3), Value(0.2), Value(0.1)],
]

out_seq = block(x_seq)
```

## Research Papers Used

This implementation is an educational adaptation of ideas from these papers and references:

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) - BitLinear, 1-bit Transformer motivation, quantized linear layers, and training-from-scratch framing.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) - ternary `{-1, 0, 1}` weights and the BitNet b1.58 quantization direction.
- [BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs](https://arxiv.org/abs/2504.18415) - Hadamard-transformed activations and lower-bit activation quantization.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - the self-attention and transformer block structure.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Adam optimizer update rules.

The `Muon` optimizer in `optimizer.py` is based on the public reference [Muon: An optimizer for hidden layers in neural networks](https://github.com/KellerJordan/Muon), which is a project/blog reference rather than a formal research paper.

## Notes

This project favors clarity over speed. Attention, quantization, normalization, and optimization are implemented with scalar Python objects, so the code is useful for learning and experimentation rather than efficient model training.

The straight-through estimator methods in `engine.Value` allow rounded or signed forward passes while preserving identity-like gradient flow during backpropagation.
