import numpy as np

# ---------------------------------------------
# Các hàm tối ưu kinh điển dùng cho benchmark
# ---------------------------------------------

def sphere(x: np.ndarray) -> float:
    """
    Sphere function
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    """
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    """
    A = 10
    n = x.size
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function
    Global minimum: f(0,...,0) = 0
    Domain: [-5, 5]^n
    """
    a, b, c = 20, 0.2, 2 * np.pi
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / n))
    term2 = -np.exp(s2 / n)
    return float(term1 + term2 + a + np.e)
