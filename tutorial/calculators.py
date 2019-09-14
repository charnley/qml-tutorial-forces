
import training

import time
import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel
from qml.representations import generate_fchl_acsf


class FCHL19Calculator(Calculator):
    """
    ASE Calculator using operator quantum machine learning on FCHL19 (FCHL ACSF)

    """

    name = 'FCHL19QMLCalculator'
    implemented_properties = ['energy', 'forces']

    def __init__(self, parameters, representations, charges, alphas, debug=False, **kwargs):
        """

        args:
            parameters - Parameters used for generating the QML represenations
            representations - Representations of the molecules used in training
            charges - Charges of the molecules used in training
            alpahas - the trained model

        """

        super().__init__(**kwargs)

        # unpack parameters
        offset = parameters["offset"]
        sigma = parameters["sigma"]

        self.parameters = parameters["representation_parameters"]

        #
        self.alphas = alphas
        self.repr = representations
        self.charges = charges

        # Offset from training
        self.offset = offset

        # Hyper-parameters
        self.sigma = sigma

        #
        self.forces = None
        self.energy = None
        self.debug = debug

        return


    def calculate(self,
        atoms: Atoms = None,
        properties=('energy', 'forces'),
        system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            raise ValueError('No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation.')

        self.query(atoms)

        if 'energy' in properties:
            self.results['energy'] = self.energy

        if 'forces' in properties:
            self.results['forces'] = self.forces

        return


    def query(self, atoms=None):

        if self.debug:
            start = time.time()

        # kcal/mol til ev
        # kcal/mol/aangstrom til ev / aangstorm
        conv_energy = 0.0433635093659
        conv_force = 0.0433635093659

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()
        n_atoms = coordinates.shape[0]

        # Calculate representation for query molecule
        rep, drep = generate_fchl_acsf(
            nuclear_charges,
            coordinates,
            gradients=True,
            **self.parameters)

        # Put data into arrays
        Qs = [nuclear_charges]
        Xs = np.array([rep], order="F")
        dXs = np.array([drep], order="F")

        # Get kernels
        Kse = get_atomic_local_kernel(self.repr, Xs, self.charges, Qs, self.sigma)
        Ks = get_atomic_local_gradient_kernel(self.repr, Xs, dXs, self.charges, Qs, self.sigma)

        # Energy prediction
        energy_predicted = np.dot(Kse, self.alphas)[0] + self.offset
        self.energy = energy_predicted * conv_energy

        # Force prediction
        forces_predicted = np.dot(Ks, self.alphas).reshape((n_atoms, 3))
        self.forces = forces_predicted * conv_force

        if self.debug:
            end = time.time()
            print("fchl19 query {:7.3f}s {:10.3f} ".format(end-start, energy_predicted))

        return

    def get_potential_energy(self, atoms=None, force_consistent=False):

        energy = self.energy

        if energy is None:
            self.query(atoms=atoms)
            energy = self.energy

        return energy

    def get_forces(self, atoms=None):

        self.query(atoms=atoms)
        forces = self.forces

        return forces


def get_calculator(dataname, debug=False):

    offset, sigma, alphas, Q, X, rep_parameters = training.read_model(dataname=dataname)

    parameters = {}
    parameters["offset"] = offset
    parameters["sigma"] = sigma
    parameters["representation_parameters"] = rep_parameters

    # set calculate class from data
    calculator = FCHL19Calculator(parameters, X, Q, alphas, debug=debug)

    return calculator


def main():

    return

if __name__ == '__main__':
    main()

