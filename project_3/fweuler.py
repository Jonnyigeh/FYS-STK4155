import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class FWEuler():
    def __init__(self, f):
        self.f = f
        self.f0 = lambda x: np.sin(np.pi * x)


    def solve(self, dx = 0.01, T = 1, L = 1):
        dt = 0.5 * dx ** 2          # Stability criterion
        N = int(L / dx)

        t = np.linspace(0, T, int(T/dt))
        x = np.linspace(0, L, N)
        u = np.zeros((len(t), N))

        u[0,:] = self.f0(x)
        u[:,0], u[:, -1] = 0, 0

        for i in range(len(t)-1):
            dudt = np.zeros_like(u[i])
            for n in range(1, N-1): # End points are constant (fixed) = 0
                dudt[n] = 1 / (dx **2) * (u[i, n+1] - 2 * u[i, n] + u[i, n-1])


            u[i+1] = u[i] + dt * dudt
        return u, t, x
    
if __name__ == "__main__":
    solver = FWEuler(lambda x, t: x ** 2)
    u, t, x = solver.solve()
    plt.plot(x, u[0],"r",label="Initial state")
    for i in range(10, len(t))[::100]:
        plt.plot(x, u[i], "b")
    plt.plot(x, u[-1], "k", label="Final state")
    plt.legend()
    plt.show()
    # breakpoint()
