import sys, os
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from ai_project.swarm.firefly import Firefly
from ai_project.problems.continuous import sphere, rastrigin, ackley
import matplotlib.pyplot as plt


# Test với hàm Sphere   
fa = Firefly(
    func=sphere, dim=30, n=50, iters=1000,
    alpha=0.25,       # giảm nhiễu
    beta0=1.5,        # tăng độ hấp dẫn
    gamma=0.5,        # mở rộng tầm nhìn
    lb=-5.12, ub=5.12,
    seed=42
)
best_x, best_f = fa.optimize()
print("Sphere:", best_f)

# Test với hàm Rastrigin
fa = Firefly(
    func=rastrigin, dim=10, n=60, iters=1500,
    alpha=0.25, beta0=1.8, gamma=0.4,
    lb=-5.12, ub=5.12, seed=7
)
best_x, best_f = fa.optimize()
print("Rastrigin:", best_f)

plt.plot(fa.history)
plt.title("Firefly Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()