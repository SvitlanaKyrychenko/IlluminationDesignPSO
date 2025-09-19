from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple
import numpy as np

from numpy import ndarray, dtype

from pso_cost_function import PSOCostFunction


class OptimizeMode(Enum):
    MIN = 0
    MAX = 1


@dataclass
class PSOConfig:
    n_particles: int = 30
    n_iters: int = 500
    c0: float = 0.95
    c1: float = 5
    c2: float = 5


class PSO:

    low_bound = float(0.0)
    high_bound = float(1.0)

    def __init__(
        self,
        config: PSOConfig = PSOConfig(),
    ):
        self.cost_function = None
        self.curr_ind = None
        self.wavelength = None
        self.config = config
        self.global_pos = None
        self.local_cost = None
        self.local_pos = None
        self.velocities = None
        self.positions = None
        self.rand = None
        self.led = None
        self.sample2 = None
        self.sample1 = None
        self.mode = None
        self.global_cost = None


    # lad shape: (10, 800N)
    # sample shape (800)
    def run_pso(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                wavelength: [Any, dtype[float]], led: ndarray[Any, dtype[float]],
                cost_function: PSOCostFunction, mode: OptimizeMode = OptimizeMode.MIN) \
            -> Tuple[ndarray[Any, dtype[float]], float]:
        if sample1.shape != sample2.shape:
            raise ValueError("samples must have the same length.")
        self.mode = mode
        self.cost_function = cost_function
        self.sample1 = sample1
        self.sample2 = sample2
        self.led = led
        self.rand = np.random.default_rng()
        self.wavelength = wavelength
        self.curr_ind = 0

        self.init_pso()
        for i in range(self.config.n_iters):
            self.step_pso()

        return self.global_pos, self.global_cost


    def init_pso(self):
        n_led = self.led.shape[0]

        self.positions = self.rand.uniform(self.low_bound, self.high_bound, size=(self.config.n_particles, n_led)).astype(np.float64)
        self.apply_constrains()
        self.velocities = self.rand.uniform(self.low_bound, self.high_bound, size=(self.config.n_particles, n_led)).astype(np.float64)
        self.local_pos = self.positions.copy()
        self.local_cost = self.calculate_cost()

        self.global_pos = self.rand.uniform(self.low_bound, self.high_bound, size=n_led)
        self.global_cost = self.worst_cost_value()


    def step_pso(self):
        r1 = self.rand.uniform(self.low_bound, self.high_bound, size=self.config.n_particles)
        r2 = self.rand.uniform(self.low_bound, self.high_bound, size=self.config.n_particles)

        # Velocity & position update
        t = self.global_pos - self.positions
        self.velocities = (self.config.c0 * self.velocities + (self.config.c1 * r1)[:, None] * (self.local_pos - self.positions) +
                           (self.config.c2 * r2)[:, None] * (self.global_pos - self.positions))
        self.positions = self.positions + self.velocities
        self.apply_constrains()

        # Evaluate
        curr_pos_cost = self.calculate_cost()

        # Update local bests
        local_pos_mask = self.find_best_cost(curr_pos_cost, self.local_cost)

        self.local_pos[local_pos_mask] = self.positions[local_pos_mask]
        self.local_cost[local_pos_mask] = curr_pos_cost[local_pos_mask]

        local_best_idx = int(np.argmin(self.local_cost))
        local_best_cost = float(self.local_cost[local_best_idx])
        curr_best_cost = self.find_best_cost(local_best_cost, self.global_cost)

        if curr_best_cost != self.global_cost:
            self.global_pos = self.positions[local_best_idx]
            self.global_cost = local_best_cost

        # Debuging
        print(f'Run {self.curr_ind + 1}/{self.config.n_iters}, cost {self.global_cost}')
        print(self.global_pos)
        self.curr_ind += 1


    def apply_constrains(self):
        self.positions = np.clip(self.positions, 0.0, None)
        self.positions = np.array([x /sum(x) for x in self.positions])


    def calculate_cost(self) -> ndarray[Any, dtype[float]]:
        if self.sample1.shape != self.sample2.shape:
            raise ValueError("samples must have the same length.")

        simulated_illuminants = np.array([pos @ self.led for pos in self.positions])
        costs = self.cost_function.calculate_cost(self.sample1, self.sample2, self.wavelength, simulated_illuminants)

        # Check for 0 vectors
        zero_mask = np.all(self.positions == 0, axis=1)
        costs[zero_mask] = self.worst_cost_value()

        return costs


    def worst_cost_value(self) -> float:
        if self.mode == OptimizeMode.MIN:
            return float("inf")
        else:
            return float("-inf")


    def find_best_cost(self, cost1, cost2) -> float:
        if self.mode == OptimizeMode.MIN:
            return cost1 < cost2
        else:
            return cost1 > cost2
