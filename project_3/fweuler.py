import numpy as np
import matplotlib.pyplot as plt
import time

def exact_solution(x,t):
    """Exact analytical solution to the 1D diffusion equation

    args:
        x           (float): Position on the rod
        t           (float): Time

    returns:
        Function value at position x, for time t
    """
    return np.sin(np.pi * x) * np.exp(- np.pi **2 * t)

class FWEuler():
    def __init__(self):
        self.f0 = lambda x: np.sin(np.pi * x)


    def solve(self, dx = 0.01, dt = None, T = 1, L = 1):
        if dt == None:
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
    solver = FWEuler()
    ## For dx = 0.01, produces comparison plot FE vs analytical at two timepoints
    if False:
        ct1 = time.perf_counter()
        u, t, x = solver.solve()
        ct2 = time.perf_counter()
        print(f"Time spent: {ct2-ct1:.6f} s")
        t1_ind = int(len(t) / 12)
        t2_ind = 4 * t1_ind
        t1 = t[t1_ind]
        t2 = t[t2_ind]
        plt.plot(x, u[0],"r", x, exact_solution(x, 0), "k--")
        plt.plot(x, u[t1_ind],"r", x, exact_solution(x, t1), "k--")
        plt.plot(x, u[t2_ind], "r", x, exact_solution(x, t2), "k--")
        plt.title("Forward Euler PDE solver with dx = 0.01")
        plt.legend(["Forward Euler", "Exact analytical"])
        plt.xlabel("Position of rod")
        plt.ylabel("Temperature of rod")
        # plt.savefig("FE_vs_anal_001.pdf")
        plt.show()

    ## dx = 0.01, produces the unstable solver presented in results
    if False:
        ct1 = time.perf_counter()
        u, t, x = solver.solve(dt = 0.01 ** 2 * 0.6)
        ct2 = time.perf_counter()
        print(f"Time spent: {ct2-ct1:.6f} s")
        plt.plot(x, u[0],"r",label=r"Initial state: $t = 0$")
        for i in range(10, len(t))[::300]:
            plt.plot(x, u[i], "k--")
        plt.plot(x, u[-1], "b", label=r"Final state: $t = t_{final}$")
        plt.title("Solution to the Heat Diffusion equation with stability criterion not met")
        plt.xlabel("Position of rod")
        plt.ylabel("Temperature of rod")
        plt.legend()
        plt.savefig("FE_stabcrit_not_met.pdf")
        plt.show()

    ## For dx = 0.1, produces comparison plot FE vs analytical at two timepoints
    if False:
        time1 = time.perf_counter()
        u, t, x = solver.solve(dx = 0.1)
        time2 = time.perf_counter()
        print(f"Time spent: {time2-time1:.6f} s")
        t1_ind = int(len(t) / 12)
        t2_ind = 4 * t1_ind
        t1 = t[t1_ind]
        t2 = t[t2_ind]
        breakpoint()
        plt.plot(x, u[0],"r", x, exact_solution(x, 0), "k--")
        plt.plot(x, u[t1_ind],"r", x, exact_solution(x, t1), "k--")
        plt.plot(x, u[t2_ind], "r", x, exact_solution(x, t2), "k--")
        plt.title("Forward Euler PDE solver with dx = 0.1")
        plt.legend(["Forward Euler", "Exact analytical"])
        plt.xlabel("Position of rod")
        plt.ylabel("Temperature of rod")
        # plt.savefig("FE_vs_anal_01.pdf")
        plt.show()
