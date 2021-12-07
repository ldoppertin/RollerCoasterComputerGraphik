import csv
import typing
from typing import *

import numpy as np

from Modules.MathHelpers import get_bezier_curve, get_bezier_first_derivative, get_bezier_second_derivative, \
    get_bezier_third_derivative


def read_csv(samples_csv_path: str) -> np.ndarray:
    """
    Reads the csv file and returns a numpy array.
    :param samples_csv_path: The path to the csv file.
    :return: The samples as np array.
    """
    sample_points = []
    with open(samples_csv_path) as samples_csv:
        csv_reader = csv.reader(samples_csv, delimiter=",")
        for sample in csv_reader:
            sample_points.append(np.array([sample[0], sample[1], sample[2]], dtype=float))
    return np.asarray(sample_points)


def read_track_file(samples_trk_path: str) -> np.ndarray:
    """
    Reads the trk file and returns a numpy array.
    Skips the "track"- and "n-samples"-keyword as well as commented line with a "#".
    :param samples_trk_path: The path to the trk file.
    :return: The samples as np array.
    """
    sample_points = []
    with open(samples_trk_path) as samples_txt:
        last_row_is_track_keyword = False
        for line in samples_txt.readlines():
            if str(line).lower().startswith("track"):
                last_row_is_track_keyword = True
            elif last_row_is_track_keyword:
                last_row_is_track_keyword = False
            elif str(line[0]) != "#" and line != "\n":
                sample_coords = []
                coord = ""
                for element in line:
                    if (element == " " or element == "\n" or element == "\t") and coord != "":
                        sample_coords.append(float(coord))
                        coord = ""
                    elif element != " " and element != "\t":
                        coord += element
                if sample_coords:
                    sample_points.append(np.array(sample_coords, dtype=float))
    return np.asarray(sample_points)


def read_file(file_path: str) -> np.ndarray:
    """
    Read a file either if type csv or trk.
    :param file_path: The file path
    :return: The samples as np array.
    """
    if file_path.endswith(".csv"):
        data = read_csv(file_path)
        return data
    elif file_path.endswith(".trk"):
        data = read_track_file(file_path)
        return data


def generate_samples_and_handles(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create p0s to p3s (startpoints p0, endpoints p3 and handlepoints p1/p2).
    :param samples: All samples.
    :return: P0s to P3s as tuple.
    """
    p1s, p2s = generate_handle_points(samples)
    p3s = np.array([(samples[(i + 1) % len(samples)]) for i in range(len(samples))])
    return samples, p1s, p2s, p3s


def generate_handle_points(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the handle points (a, b) for each sample.
    :param samples: The samples.
    :return: The handle points for each sample: an "A" np array and a "B" np array
    """
    n_samples = len(samples)
    coefficients = generate_coefficients(n_samples)
    rhs = np.array([(4 * samples[(i + 1) % n_samples] + 2 * samples[(i + 2) % n_samples]) for i in range(n_samples)])

    # calculate the a's and b's ("handles") for each sample
    a_aka_p1 = np.array(np.linalg.solve(coefficients, rhs), dtype=float)
    b_aka_p2 = np.array([(2 * samples[(i + 1) % n_samples] - a_aka_p1[(i + 1) % n_samples]) for i in range(n_samples)],
                        dtype=float)
    return a_aka_p1, b_aka_p2


def generate_coefficients(n_samples: int) -> np.ndarray:
    """
    Generate the cubic bezier coefficients for n samples. The values were calculated on paper and are essential for
    making the bezier-curves closed, "knick-frei" and "kruemmungsstetig".
    :param n_samples: The number of samples.
    :return: The coefficients as np array.
    """
    # in the beginning, all coefficients are 0
    coefficients = np.zeros((n_samples, n_samples), dtype=float)

    for i, row in enumerate(coefficients):
        # all rows are 1, 4, 1 at ai, ai+1 and ai+2.
        # if i + 1 or i + 2 > n_samples, the first ones are chosen
        a_plus_1 = (i + 1) % n_samples
        a_plus_2 = (i + 2) % n_samples
        row[i] = 1
        row[a_plus_1] = 4
        row[a_plus_2] = 1
    return coefficients


def get_curve_data(samples: np.ndarray) -> Dict[str, typing.List]:
    """
    Get the basic curve data: beziers and their derivatives as lambda functions.
    :param samples: the samples (points)
    :return: a data dictionary
    """
    data = {"beziers": [], "beziers_first_derivative": [], "beziers_second_derivative": [],
            "beziers_third_derivative": []}
    p0s, p1s, p2s, p3s = generate_samples_and_handles(samples)
    for i in range(len(samples)):
        bezier = get_bezier_curve(p0s[i], p1s[i], p2s[i], p3s[i])
        data["beziers"].append(bezier)
        bezier_first_derivative = get_bezier_first_derivative(p0s[i], p1s[i], p2s[i], p3s[i])
        data["beziers_first_derivative"].append(bezier_first_derivative)
        bezier_second_derivative = get_bezier_second_derivative(p0s[i], p1s[i], p2s[i], p3s[i])
        data["beziers_second_derivative"].append(bezier_second_derivative)
        bezier_third_derivative = get_bezier_third_derivative(p0s[i], p1s[i], p2s[i], p3s[i])
        data["beziers_third_derivative"].append(bezier_third_derivative)
    return data
