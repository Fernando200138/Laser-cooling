import time

import numpy as np
import scipy.constants as cts



from model.molecules  import Molecule, BaF


def force_to_acceleration(forces: np.ndarray, molecule: Molecule) -> np.ndarray:
    k = 2 * cts.pi / molecule.wave_length

    return forces * (molecule.line_width_in_MHz * 1e6) * 2 * cts.pi * cts.hbar * k / molecule.mass


def velocity_to_si(velocity: np.ndarray, molecule: Molecule) -> np.ndarray:
    k = 2 * cts.pi / molecule.wave_length

    return velocity * ((molecule.line_width_in_MHz * 1e6) * 2 * cts.pi / k)


def velocity_from_si(velocity_in_si: np.ndarray, molecule: Molecule) -> np.ndarray:
    k = 2 * cts.pi / molecule.wave_length

    return velocity_in_si / ((molecule.line_width_in_MHz * 1e6) * 2 * cts.pi / k)


def use_symmetry_force_curve(velocities: np.ndarray, forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.concatenate((-np.flip(velocities)[:-1], velocities))
    forces = np.concatenate((-np.flip(forces)[:-1], forces))

    return velocities, forces


def filter_points(velocities, forces, points) -> tuple[list, list]:
    wrong_velocities = [velocities[point] for point in points]
    wrong_forces = [forces[point] for point in points]

    return wrong_velocities, wrong_forces


def get_estimated_end_time(time_start: float, iteration: int, total_iterations: int, format: str = '%H:%M:%S') -> str:
    time_taken = time.time() - time_start
    estimated_time_end_seconds = time_taken / (iteration + 1) * total_iterations + time_start + 3600
    return time.strftime(format, time.gmtime(estimated_time_end_seconds))

