
import numpy as np
import pandas as pd
import ast

import qml
from qml.math import svd_solve
from qml.representations import generate_fchl_acsf
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_atomic_local_gradient_kernel


FILENAMES = [
    "mc_amon00000_001_C.csv",
    "mc_amon00001_001_O.csv",
    "mc_amon00003_001_C=O.csv",
    "mc_amon00005_001_CO.csv",
    "mc_amon00007_001_CC.csv",
    "mc_amon00011_001_C=C.csv",
    "mc_amon00012_001_CC=O.csv",
    "mc_amon00016_001_C=CO.csv",
    "mc_amon00024_001_COC.csv",
    "mc_amon00030_001_CCO.csv",
]

DISTANCE_CUT = 4.0

DEFAULT_PARAMETERS = {
    "rcut": DISTANCE_CUT,
    "acut": DISTANCE_CUT,
}

DEFAULT_ELEMENTS = [1, 6, 8]

MAX_ATOMS = 25

# for tutorial sake
np.random.seed(42)

def read_csv_file(filename, n=32,
    parameters=DEFAULT_PARAMETERS,
    elements=DEFAULT_ELEMENTS,
    max_atoms=MAX_ATOMS):
    """

    """

    df = pd.read_csv(filename, sep=";")

    max_n = len(df["atomization_energy"])
    n = min(max_n, n)

    random_indexes = np.random.choice(max_n, size=n, replace=False)

    representations = []
    d_representations = []
    charges = []

    energies = []
    forces = []

    for i in random_indexes:

        # atomistic
        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        # properties
        force = np.array(ast.literal_eval(df["forces"][i]))
        energy = float(df["atomization_energy"][i])

        # calculate representations
        rep, drep = generate_fchl_acsf(
                nuclear_charges,
                coordinates,
                gradients=True,
                pad=max_atoms,
                elements=elements,
                **parameters)

        #
        representations.append(rep)
        d_representations.append(drep)
        charges.append(nuclear_charges)
        energies.append(energy)
        forces.append(force)


    return representations, d_representations, charges, energies, forces


def read_csv_files(filenames, **kwargs):

    representations = []
    d_representations = []
    charges = []
    energies = []
    forces = []

    for filename in filenames:
        x1, dx1, q1, e1, f1 = read_csv_file(filename, **kwargs)

        representations += x1
        d_representations += dx1
        charges += q1
        energies += e1
        forces += f1

    representations = np.asarray(representations)
    d_representations = np.asarray(d_representations)
    charges = np.asarray(charges)
    energies = np.asarray(energies)
    forces = np.asarray(forces)

    return representations, d_representations, charges, energies, forces


def generate_kernel(X1, X2, dX, charges1, charges2, sigma=10.0, **kwargs):
    """
    x representations
    dx d_representations
    """

    Kte = get_atomic_local_kernel(X1, X2,  charges1, charges2,  sigma)
    Kt = get_atomic_local_gradient_kernel(X1, X2,  dX,  charges1, charges2, sigma)

    return Kte, Kt


def training(kernel_te, kernel_t, Y_te, Y_t, sigma=10.0):
    """
    """

    C = np.concatenate((kernel_te, kernel_t))
    Y = np.concatenate((Y_te, Y_t.flatten()))

    alpha = svd_solve(C, Y, rcond=1e-11)

    return alpha


def learning_curve(filenames, sigma=10):

    representations, d_representations, charges, energies, forces = read_csv_files(filenames)

    n_points = len(energies)
    indexes = np.arange(n_points, dtype=int)
    np.random.shuffle(indexes)

    print("total n points", n_points)

    n_valid = 50
    validation_idx = indexes[-n_valid:]

    validation_repr = representations[validation_idx]
    validation_d_repr = d_representations[validation_idx]
    validation_charges = [charges[i] for i in validation_idx]
    validation_energies = energies[validation_idx]
    validation_forces = [forces[i] for i in validation_idx]
    validation_forces = np.concatenate(validation_forces)

    n_training = [2**x for x in range(1, 8)]

    for n in n_training:

        training_indexes = indexes[:n]
        training_repr = representations[training_indexes]
        training_d_repr = d_representations[training_indexes]
        training_charges = [charges[i] for i in training_indexes]
        training_energies = energies[training_indexes]
        training_forces = [forces[i] for i in training_indexes]
        training_forces = np.concatenate(training_forces)

        # Generate kernel
        kernel_te, kernel_t = generate_kernel(
                training_repr,
                training_repr,
                training_d_repr,
                training_charges,
                training_charges,
                sigma=sigma)

        alpha = training(kernel_te, kernel_t, training_energies, training_forces, sigma=sigma)

        # Generate kernel
        kernel_pte, kernel_pt = generate_kernel(
                training_repr,
                validation_repr,
                validation_d_repr,
                training_charges,
                validation_charges,
                sigma=sigma)

        energy_predicted = np.dot(kernel_pte, alpha)

        error = energy_predicted - validation_energies

        print("{:5d}".format(n), "{:10.2f}".format(np.abs(error).mean()))


    return


def read_model(dataname="_deploy_"):

    offset = np.load("data/"+dataname+"_offset.npy")
    sigma = np.load("data/"+dataname+"_sigma.npy")
    alpha = np.load("data/"+dataname+"_alphas.npy")
    charges = np.load("data/"+dataname+"_Q.npy", allow_pickle=True)
    representations = np.load("data/"+dataname+"_X.npy")

    return offset, sigma, alpha, charges, representations


def deploy(filenames, n_train=100, sigma=10.0, dataname="_deploy_"):

    offset = 0.0

    representations, d_representations, charges, energies, forces = read_csv_files(filenames)

    n_points = len(energies)

    if n_points < n_train:
        print("not enough training points", n_train, ">", n_points)
        quit()

    indexes = list(range(n_points))
    np.random.shuffle(indexes)
    training_indexes = indexes[:n_train]

    training_repr = representations[training_indexes]
    training_d_repr = d_representations[training_indexes]
    training_charges = [charges[i] for i in training_indexes]
    training_energies = energies[training_indexes]
    training_forces = [forces[i] for i in training_indexes]
    training_forces = np.concatenate(training_forces)

    kernel_te, kernel_t = generate_kernel(training_repr, training_repr, training_d_repr, training_charges, training_charges, sigma=sigma)

    alpha = training(kernel_te, kernel_t, training_energies, training_forces)

    # Save results
    if dataname is not None:
        np.save("data/"+dataname+"_offset.npy", offset)
        np.save("data/"+dataname+"_sigma.npy", sigma)
        np.save("data/"+dataname+"_alphas.npy", alpha)
        np.save("data/"+dataname+"_Q.npy", charges, allow_pickle=True)
        np.save("data/"+dataname+"_X.npy", representations)

    return offset, sigma, alpha, charges, representations


def test():

    # define chemical space
    directory = "data/training_data/"
    filenames = [directory + f for f in FILENAMES]

    # learning curves on random points
    learning_curve(filenames)

    return


def main():

    # define chemical space
    directory = "data/training_data/"
    filenames = [directory + f for f in FILENAMES]

    deploy(filenames, dataname="_deploy_")

    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test()

    else:
        main()

