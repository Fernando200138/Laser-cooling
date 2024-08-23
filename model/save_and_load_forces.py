from typing import Optional

import numpy as np
#import pandas as pd

from model.helpers import velocity_to_si, force_to_acceleration
from model.molecules import Molecule
from model.molecules import BaF
molecule=BaF

def save_forces(velocities: np.ndarray, forces: np.ndarray, method: str, detuning: float, saturation: float,
                molecule: Molecule,
                velocity_in_y: bool = False,
                additional_title: str = '',
                directory: str = 'data'
                ):
    velocity_force_array = np.stack((velocities, forces), axis=1)
    title=make_title(method, detuning, saturation, molecule.get_name(), additional_title, velocity_in_y, directory)
    print(title)
    np.savetxt(title,velocity_force_array,fmt='%.4e', delimiter=',')


def load_forces(method: str, detuning: float, saturation: float, molecule: Molecule, velocity_in_y: bool = False,
                additional_title: str = '', directory: str = 'data') -> \
        Optional[tuple[np.ndarray, np.ndarray]]:
    title = make_title(method, detuning, saturation, molecule.get_name(), additional_title, velocity_in_y, directory)
    try:
        velocity_force_array = np.genfromtxt(title, delimiter=',')
    except:
        print('No data found for %s' % title)
        return
    #print(title)
    velocities = velocity_to_si(velocity_force_array[:, 0], molecule)
    forces = force_to_acceleration(velocity_force_array[:, 1], molecule)
    return velocities, forces


'''
def load_forces_2(method: str, detuning: float, saturation: float, molecule: Molecule, velocity_in_y: bool = False,
                additional_title: str = '', directory: str = 'data') -> \
        Optional[tuple[np.ndarray, np.ndarray]]:
    title = make_title(method, detuning, saturation, molecule.get_name(), additional_title, velocity_in_y, directory)
    try:
        data=
'''

def make_title(method: str, detuning: float, saturation: float, molecule_name: str,
               additional_title: Optional[str] = None,
               velocity_in_y: bool = False, directory:str = 'data') -> str:
    return '../data/%s/%s_%.2f_%.2f_%s%s%s.txt' % ( #I had to edit this line it used to say '../data/%s/%s_%.2f_%.2f_%s%s%s.txt' but it didn't work
        directory, method, detuning, saturation, molecule_name, '_' + additional_title if additional_title else '',
        '_y' if velocity_in_y else '')


def load_forces_from_tarbut(power) -> tuple[np.ndarray, np.ndarray]:
    title = '../data/data_from_tarbut/-0.64_%.0f' % power
    velocity_force_array = np.genfromtxt(title, delimiter=',')

    return velocity_force_array[:, 0], velocity_force_array[:, 1]


