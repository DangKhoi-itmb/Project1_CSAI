import numpy as np

class Firefly:
    def __init__(self, func, dim, n = 30, iters = 300, alpha = 0.25, beta0 = 1.5, gamma = 0.5, lb = -5.12, ub = 5.12, seed = None):
        self.func = func
        self.dim = dim
        self.n = n
        self.iters = iters
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.lb = lb
        self.ub = ub
        self.rng = np.random.default_rng(seed)
        self.history = []
    
    def _init_population(self):
        X = self.rng.uniform(self.lb, self.ub, (self.n, self.dim))
        F = np.apply_along_axis(self.func, 1, X)
        return X, F
    
    def _move(self, xi, xj):
        r = np.linalg.norm(xi - xj)
        beta = self.beta0 * np.exp((-1) * self.gamma * r * r)
        step = beta * (xj - xi) + self.alpha * (self.rng.random(self.dim) - 0.5)
        x_new = xi + step
        x_new = np.clip(x_new, self.lb, self.ub)
        return x_new
    
    def optimize(self):
        X, F = self._init_population()
        best_idx = np.argmin(F)
        best_x, best_f = X[best_idx].copy(), F[best_idx]
        
        alpha0 = self.alpha
        for t in range (self.iters):
            # 🔹 giảm α dần theo exp decay
            self.alpha = alpha0 * (0.97 ** t)
            # có thể thay đổi gamma, beta0 theo iteration (tuỳ chọn)
            self.gamma = 0.5 + 0.5 * (t / self.iters)     # từ 0.5 → 1.0
            self.beta0 = 1.5 - 0.5 * (t / self.iters)     # từ 1.5 → 1.0
            for i in range (self.n):
                for j in range (self.n):
                    if F[j] < F[i]:
                        X[i] = self._move(X[i], X[j])
                        F[i] = self.func(X[i])
                        
            bi = np.argmin(F)
            if F[bi] < best_f:
                best_f = F[bi]
                best_x = X[bi].copy()
            self.history.append(best_f)
            
        return best_x, best_f