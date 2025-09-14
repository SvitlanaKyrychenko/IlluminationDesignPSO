import numpy as np
from pso import PSOConfig, PSO, OptimizeMode, CostFunctionMode


# Test on random data
def run_random_pso():
    rand = np.random.default_rng()
    low_bound = float(0.0)
    high_bound = float(1.0)
    n_particles = 500
    n_wavelength = 200

    pso_config = PSOConfig(30, n_particles, 0.95, 5, 5)
    pso = PSO(pso_config)

    sample1 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    sample2 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    wavelength = np.arange(400, 400 + n_wavelength*2, 2, dtype=float)

    led = rand.uniform(low_bound, high_bound, size=(10, n_wavelength)).astype(np.float64)
    global_pos, global_cost = pso.run_pso(sample1, sample2, wavelength, led, CostFunctionMode.CIEDE, OptimizeMode.MIN)
    print(global_pos)
    print(global_cost)


if __name__ == '__main__':
    run_random_pso()
