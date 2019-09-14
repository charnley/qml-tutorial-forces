
import calculators
import rmsd

import ase
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import BFGS

def dump_xyz(molecule, filename):

    coordinates = molecule.get_positions()
    nuclear_charges = molecule.get_chemical_symbols()
    txt = rmsd.set_coordinates(nuclear_charges, coordinates)

    with open(filename, 'w') as f:
        f.write(txt)

    return


def main():

    calculator = calculators.get_calculator("_deploy_", debug=False)
    atom_labels, coordinates = rmsd.get_coordinates_xyz("examples/ethanol.xyz")

    molecule = ase.Atoms(atom_labels, coordinates)
    molecule.set_calculator(calculator)

    energy = molecule.get_potential_energy()
    print(energy)

    dyn = BFGS(molecule)
    dyn.run(fmax=0.5)

    dump_xyz(molecule, "_tmp_molecule_optimize.xyz")

    return


if __name__ == '__main__':
    main()

