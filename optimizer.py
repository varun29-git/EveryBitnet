import math

class Adam:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad == 0: continue
            
            # Update biased first and second moment estimates
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad**2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.betas[0]**self.t)
            v_hat = self.v[i] / (1 - self.betas[1]**self.t)
            
            # Update parameter
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

class Muon:
    """
    A 'First Principles' implementation of the Muon Optimizer.
    Muon uses Newton-Schulz iteration to perform orthogonal updates.
    """
    def __init__(self, parameters, lr=0.02, momentum=0.95, ns_steps=5):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.m = [0.0] * len(parameters)

    def step(self):
        # In this scalar engine, Muon is simplified to act on the momentum 
        # of each parameter, applying an 'orthogonal-like' scaling.
        # True Muon requires matrix-level SVD/Newton-Schulz.
        
        for i, p in enumerate(self.parameters):
            if p.grad == 0: continue
            
            # 1. Update momentum
            self.m[i] = self.momentum * self.m[i] + p.grad
            
            # 2. Newton-Schulz Iteration (Simplified for scalar 'representation' learning)
            # In a matrix engine, we'd do: X = X * (1.5*I - 0.5 * X^T * X)
            # For a scalar, 'orthogonal' simply means maintaining unit scale
            update = self.m[i]
            for _ in range(self.ns_steps):
                # Scalar version of Newton-Schulz: x = x * (1.5 - 0.5 * x^2)
                # This drives the update towards unit magnitude (orthogonal in 1D)
                update = update * (1.5 - 0.5 * update**2)
            
            # 3. Apply update
            p.data -= self.lr * update

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
