import math

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0
    # The last output grad will be made 1, to run the operation.
    self._backward = lambda : None
    self.label = label
    '''
    1. _children=() is an empty tuple that stores which Value objects were used
    to calculate this value object.
    2. _op shows you how the values in children were used to get this value.
    EX:
    C = Value(A*B) where A and B are previously defined values
    '''
  def __repr__(self):
    return f"Value(data={self.data})"
    # Returns value of the stored data whenever the print is called.
    # It tells python how to represent the data whenever asked for.

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+") # Returning a new Value Object.
    def _backward():
      self.grad += 1.0  * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
    # Without this if we add it with another value it will show an error.

  def __radd__(self, other):
    return self + other

  def rmul(self, other):
    return self * other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
    # Without this if we multiple it with another value it will show an error.
  def __truediv__(self,other):
    return self * other**-1

  def __neg__(self):
    return self * -1

  def __sub__(self,other):
    return self + (-other)

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')
    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out =  Value(t, (self, ), 'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
  def backward(self):
    # We have automated the backpropagation, but we will have to run it manually for each element
    # We will use a TOPOLOGICAL GRAPH to solve this problem
    topo = []
    visited = set()
    def _build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          _build_topo(child)
        topo.append(v)
    _build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def ste_sign(self):
      """Straight-Through Estimator for Sign function."""
      out = Value(1.0 if self.data >= 0 else -1.0, (self,), 'ste_sign')
      def _backward():
          # Identity gradient flow
          self.grad += 1.0 * out.grad
      out._backward = _backward
      return out

  def ste_round(self):
      """Straight-Through Estimator for Round function."""
      out = Value(round(self.data), (self,), 'ste_round')
      def _backward():
          # Identity gradient flow
          self.grad += 1.0 * out.grad
      out._backward = _backward
      return out