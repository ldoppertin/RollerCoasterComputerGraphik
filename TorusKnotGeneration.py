"""
PLEASE SPECIFY P, Q AND THE NUMBER OF SAMPLES OF THE TORUS KNOT YOU WANT TO SAMPLE TO RUN THIS CODE.
"""

from math import *

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-q")
parser.add_argument("-nSamples")
args = parser.parse_args()


def generate_torus_knot_samples(p: int, q: int, sample_count: int) -> np.ndarray:
    """
    Generate a sample array from a torus.
    :param p: P used for torus generation.
    :param q: Q used for torus generation.
    :param sample_count: The number of samples, sampled from the torus.
    :return: The samples as np array.
    """
    sample_points = []

    for n in range(sample_count):
        phi = (n / sample_count) * 2 * pi
        radius = cos(q * phi) + 2
        x = radius * cos(p * phi)
        y = radius * sin(p * phi)
        z = - sin(q * phi)
        sample_points.append(np.array([x, y, z]))
    return np.asarray(sample_points)


def save_as_csv(sample_array: np.ndarray, name: str) -> None:
    """
    Saves the given sample array as a csv file.
    :param sample_array: all samples
    :param name: the name of the csv file
    """
    np.savetxt(name, sample_array, delimiter=",")


if __name__ == "__main__":
    p = int(args.p)
    q = int(args.q)
    n_samples = int(args.nSamples)
    print(p, q, n_samples)
    torus_samples = generate_torus_knot_samples(p, q, n_samples)
    save_as_csv(torus_samples, f"TorusSamples_{p}_{q}_{n_samples}.csv")
    print("Script finished.")
