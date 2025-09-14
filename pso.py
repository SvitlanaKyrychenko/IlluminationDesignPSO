from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple
import numpy as np

from numpy import ndarray, dtype
from cost_functions import ciede
from utils import spim2rgb, spim2XYZ, XYZ2Lab


class CostFunctionMode(Enum):
    CIEDE = 0
    MICHELSON_CONTARST = 1
    RGBDE = 2


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
        self.cost_mode = None
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
                cost_mode: CostFunctionMode.CIEDE, mode: OptimizeMode = OptimizeMode.MIN) \
            -> Tuple[ndarray[Any, dtype[float]], float]:
        if sample1.shape != sample2.shape:
            raise ValueError("samples must have the same length.")
        self.mode = mode
        self.cost_mode = cost_mode
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
        self.velocities = self.rand.uniform(0, 0.1, size=(self.config.n_particles, n_led)).astype(np.float64)
        self.local_pos = self.positions.copy()
        self.local_cost = self.calculate_cost()

        self.global_pos = self.rand.uniform(self.low_bound, self.high_bound, size=n_led)
        if self.mode == OptimizeMode.MAX:
            self.global_cost = float("-inf")
        else:  # mode == OptimizeMode.MIN
            self.global_cost = float("inf")


    def step_pso(self):
        r1 = self.rand.uniform(self.low_bound, self.high_bound, size=self.config.n_particles)
        r2 = self.rand.uniform(self.low_bound, self.high_bound, size=self.config.n_particles)

        # Velocity & position update
        self.velocities = (self.config.c0 * self.velocities + (self.config.c1 * r1)[:, None] * (self.local_pos - self.positions) +
                           (self.config.c2 * r2)[:, None] * (self.global_pos - self.positions))
        self.positions = self.positions + self.velocities

        self.apply_constrains()

        # Evaluate
        curr_pos_cost = self.calculate_cost()

        # Update local bests
        if self.mode == OptimizeMode.MIN:
            local_pos_mask = curr_pos_cost < self.local_cost
        else:
            local_pos_mask = curr_pos_cost > self.local_cost

        self.local_pos[local_pos_mask] = self.positions[local_pos_mask]
        self.local_cost[local_pos_mask] = curr_pos_cost[local_pos_mask]

        local_best_idx = int(np.argmin(self.local_cost))
        local_best_cost = float(self.local_cost[local_best_idx])
        if self.mode == OptimizeMode.MIN:
            if local_best_cost < self.global_cost:
                self.global_pos = self.positions[local_best_idx]
                self.global_cost = local_best_cost
        else:
            if local_best_cost > self.global_cost:
                self.global_pos = self.positions[local_best_idx]
                self.global_cost = local_best_cost
        # Debuging
        print(f'Run {self.curr_ind + 1}/{self.config.n_iters}, cost {local_best_cost}')
        print(self.global_pos)
        self.curr_ind += 1


    def apply_constrains(self):
        self.positions = np.clip(self.positions, 0.0, None)
        self.positions = np.array([x /sum(x) for x in self.positions])


    def calculate_cost(self) -> ndarray[Any, dtype[float]]:
        if self.sample1.shape != self.sample2.shape:
            raise ValueError("samples must have the same length.")
        simulated_illuminant = self.positions[:, :, None] * self.led[None, :, :]
        corrected_sample1 = (simulated_illuminant * self.sample1[None, None, :]).sum(axis=1)
        corrected_sample2 = (simulated_illuminant * self.sample2[None, None, :]).sum(axis=1)

        if self.cost_mode == CostFunctionMode.CIEDE:
            sample1_xyz = np.array([spim2XYZ(x, self.wavelength) for x in corrected_sample1])
            sample2_xyz = np.array([spim2XYZ(x, self.wavelength) for x in corrected_sample2])
            sample1_lab = np.array([np.squeeze(XYZ2Lab(x))for x in sample1_xyz])
            sample2_lab = np.array([np.squeeze(XYZ2Lab(x))for x in sample2_xyz])
            costs = np.array([
                ciede(s1, s2) for s1, s2 in zip(sample1_lab, sample2_lab)
            ])

        # Check for 0 vectors
        zero_mask = np.all(self.positions == 0, axis=1)
        costs[zero_mask] = np.inf

        return costs
