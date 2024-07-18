import scipy.constants as cts

ATOMIC_MASS_CONSTANT = cts.value('atomic mass constant')

"""
This file contains molecule representations used in the NL-eEDM BaF experiment. Dont use PyLCP in this file, 
in order to separate responsibilities 

By Max Vos 19/02/2024 
"""


class Molecule:
    def __init__(self,
                 name: str,
                 line_width: float = 0.,
                 nuclear_spin: float = 1 / 2,
                 ground_N: int = 1,
                 ground_B: float = 0.,
                 ground_gamma: float = 0.,
                 ground_b: float = 0.,
                 ground_c: float = 0.,
                 excited_J: float = 1 / 2,
                 excited_B: float = 0.,
                 excited_parity: float = 1,
                 frosch_a: float = 0.,
                 frosch_b: float = 0.,
                 frosch_c: float = 0.,
                 sigma: float = 1 / 2,
                 mass: float = 0,
                 wave_length: float = 860e-9
                 ):
        self.name = name
        """
        Line width of excited state in MHz, used for unit
        """
        self.line_width_in_MHz = line_width
        """
        Nuclear spin quantum number (for F 1/2)
        """
        self.nuclear_spin = nuclear_spin
        """
        Rotational quantum number
        """
        self.ground_N = ground_N
        """
        Rotational constant (in cm-1)
        """
        self.ground_B = ground_B
        """
        Electron-spin rotational coupling constant (MHz)
        """
        self.ground_gamma = ground_gamma
        """
        Isotropic spin-spin interaction
        """
        self.ground_b = ground_b
        """
        Anisotropic spin-spin interaction
        """
        self.ground_c = ground_c
        """
        Rotational quantum nuber
        """
        self.excited_J = excited_J
        """
        Rotational constant (in MHz)
        """
        self.excited_B = excited_B
        """
        Parity quantum number
        TODO
        """
        self.excited_parity = excited_parity
        """
        Frosch and Foley a parameter
        """
        self.frosch_a = frosch_a
        """
        Frosch and Foley b parameter
        """
        self.frosch_b = frosch_b
        """
        Frosch and Foley c parameter
        """
        self.frosch_c = frosch_c
        """
        Sigma quantum number
        """
        self.sigma = sigma
        self.mass = mass
        self.wave_length = wave_length

    def get_name(self):
        return self.name


BaF = Molecule(
    line_width=2.78,
    # from: https://webbook.nist.gov/cgi/inchi?ID=C13966706&Mask=1000
    ground_B=.2158 * cts.c / 100,
    # from: Ernst et al., J. Chem. Phys. (1986)
    ground_gamma=80.923,
    ground_b=63.509,
    ground_c=8.224,
    # from: https://webbook.nist.gov/cgi/inchi?ID=C13966706&Mask=1000
    excited_B=.2118 * cts.c / 100,
    # from: NL-eedm: Benchmarking of the Fock-space coupled-cluster ... (2022)
    frosch_a=26.55,
    name='BaF',
    mass=156.3254 * ATOMIC_MASS_CONSTANT,
    wave_length=860e-9
)

CaF = Molecule(
    line_width=8.3,
    # from: https://webbook.nist.gov/cgi/inchi?ID=C13827264&Mask=1000
    ground_B=.338 * cts.c / 100,
    # from: https://webbook.nist.gov/cgi/inchi?ID=C13827264&Mask=1000
    excited_B=.343 * cts.c / 100,
    # from pylcp CaF MOT example
    ground_gamma=39.65891,
    ground_b=109.1893,
    ground_c=40.1190,
    frosch_a=3 / 2 * 4.8,
    name='CaF',
    mass=59.0764 * ATOMIC_MASS_CONSTANT,
    # from pylcp CaF MOT example
    wave_length=606e-9
)

SrF = Molecule(
    # from: https://webbook.nist.gov/cgi/inchi?ID=C13827264&Mask=1000 Excited unknown so taken same as ground_B.
    excited_B=0.250533 * cts.c / 100,
    ground_B=0.250533 * cts.c / 100,

    # from: https://core.ac.uk/download/pdf/76988423.pdf page 37
    ground_b=97.0834,
    ground_c=30.268,
    ground_gamma=74.79485,
    name='SrF',
    mass=106.62 * ATOMIC_MASS_CONSTANT
)
