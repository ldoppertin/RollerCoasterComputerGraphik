import typing
from typing import *

import numpy as np


def get_bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Union[int, Any]:
    """
    Get the cubic bezier curve function from the p0 point to p3. P0 e.g. should have length 3 for 3 dimensions.
    :param p0: a np.array of 3 dimensions: x, y, z
    :param p1: a np.array of 3 dimensions: x, y, z
    :param p2: a np.array of 3 dimensions: x, y, z
    :param p3: a np.array of 3 dimensions: x, y, z
    :return: The function (lambda) (t: Any)  -> Union[int, Any]
    """
    return lambda t: ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def get_bezier_first_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Union[int, Any]:
    """
    Get the first derivative of cubic bezier curve function from the p0 point to p3. P0 e.g. should have length 3 for 3
    dimensions.
    :param p0: a np.array of 3 dimensions: x, y, z
    :param p1: a np.array of 3 dimensions: x, y, z
    :param p2: a np.array of 3 dimensions: x, y, z
    :param p3: a np.array of 3 dimensions: x, y, z
    :return: The function (lambda) (t: Any)  -> Union[int, Any]
    """
    return lambda t: -3 * p0 * ((1 - t) ** 2) + 3 * p1 * ((1 - t) ** 2) - 6 * p1 * (1 - t) * t + 6 * p2 * (
            1 - t) * t - 3 * p2 * (t ** 2) + 3 * p3 * (t ** 2)


def get_bezier_second_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Union[int, Any]:
    """
    Get the second derivative of cubic bezier curve function from the p0 point to p3. P0 e.g. should have length 3 for 3
    dimensions.
    :param p0: a np.array of 3 dimensions: x, y, z
    :param p1: a np.array of 3 dimensions: x, y, z
    :param p2: a np.array of 3 dimensions: x, y, z
    :param p3: a np.array of 3 dimensions: x, y, z
    :return: The function (lambda) (t: Any)  -> Union[int, Any]
    """
    return lambda t: 6 * p0 * (1 - t) - 12 * p1 * (1 - t) + 6 * p1 * t + 6 * p2 * (1 - t) - 12 * p2 * t + 6 * p3 * t


def get_bezier_third_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Union[int, Any]:
    """
    Get the second derivative of cubic bezier curve function from the p0 point to p3. P0 e.g. should have length 3 for 3
    dimensions.
    :param p0: a np.array of 3 dimensions: x, y, z
    :param p1: a np.array of 3 dimensions: x, y, z
    :param p2: a np.array of 3 dimensions: x, y, z
    :param p3: a np.array of 3 dimensions: x, y, z
    :return: The function (lambda) (t: Any)  -> Union[int, Any]
    """
    return lambda t: -6 * p0 + 12 * p1 + 6 * p1 - 6 * p2 - 12 * p2 + 6 * p3


def get_curvature(cross_product_of_first_second_derivative: np.ndarray, first_derivative: typing.List[float]):
    """
    Get the curvature.
    """
    speed = get_norm(first_derivative)
    curvature = cross_product_of_first_second_derivative / (speed ** 3)
    return np.array(curvature)


def get_torsion(cross_product_of_first_second_derivative: np.ndarray,
                third_derivative: typing.List[float]) -> np.ndarray:
    """
    Get the torsion.
    """
    torsion = (cross_product_of_first_second_derivative * third_derivative) / \
              (get_norm(cross_product_of_first_second_derivative) ** 2)
    return np.array(torsion)


def get_binormal(cross_product_of_first_second_derivative: np.ndarray) -> np.ndarray:
    """
    Get the binormals.
    """
    binormals = cross_product_of_first_second_derivative / get_norm(cross_product_of_first_second_derivative)
    return binormals


def get_normal(binormals: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """
    Get the normals.
    """
    normals = get_cross_product(binormals, tangents)
    return normals


def get_tangent(first_derivative: typing.List[float]) -> np.ndarray:
    """
    Get the tangent given the first derivative.
    """
    tangents = first_derivative / get_norm(first_derivative)
    return np.array(tangents)


def get_cross_product(first_value: Union[np.ndarray, Iterable],
                      second_value: Union[np.ndarray, Iterable]) -> np.ndarray:
    """
    Get the cross product of the two given values.
    :return: The cross product.
    """
    cross_product = np.cross(first_value, second_value)
    return cross_product


def get_norm(to_be_normed: Union[np.ndarray, Iterable]) -> np.ndarray:
    """
    Get the norm of the e.g. array-like argument.
    :return: The normed parameter.
    """
    return np.linalg.norm(to_be_normed)


def get_geometric_behaviour(curve_first_derivatives: typing.List[typing.List[float]],
                            curve_second_derivatives: typing.List[typing.List[float]],
                            curve_third_derivatives: typing.List[typing.List[float]]) -> typing.Dict:
    """
    Get the geometric behaviour of the curves from curvatures to jolts by using the points on the curves' derivatives.
    :param curve_first_derivatives: A list of x-dimensional points of the first derivative.
    :param curve_second_derivatives: A list of x-dimensional points of the second derivative.
    :param curve_third_derivatives: A list of x-dimensional points of the third derivative.
    :return: A dictionary with all the geometric information.
    """
    geometric_values = {"curvatures": [], "curvature_norms": [], "torsions": [], "torsion_norms": [], "binormals": [],
                        "tangents": [], "normals": [], "speeds": [], "accelerations": [], "jolts": []}
    for i, first_derivative in enumerate(curve_first_derivatives):
        determinant_of_first_second_derivative = get_cross_product(first_derivative, curve_second_derivatives[i])
        curvature = get_curvature(determinant_of_first_second_derivative, first_derivative)
        geometric_values["curvatures"].append(curvature)
        geometric_values["curvature_norms"].append(get_norm(curvature))
        torsion = get_torsion(determinant_of_first_second_derivative, curve_third_derivatives[i])
        geometric_values["torsions"].append(torsion)
        geometric_values["torsion_norms"].append(get_norm(torsion))
        binormal = get_binormal(determinant_of_first_second_derivative)
        geometric_values["binormals"].append(binormal)
        tangent = get_tangent(first_derivative)
        geometric_values["tangents"].append(tangent)
        geometric_values["normals"].append(get_normal(binormal, tangent))
        geometric_values["speeds"].append(first_derivative)
        geometric_values["accelerations"].append(curve_second_derivatives[i])
        geometric_values["jolts"].append(curve_third_derivatives[i])
    return geometric_values


# Attempt to get the dynamic behaviour via runge kutta. Problem: use runge kutta 2 times
"""
G = np.array([0, 0, 9.81])
MASS = 20

def get_energy(g: np.ndarray, mass: float, bezier_curve_function, bezier_curve_first_derivative):
    return lambda t: (1 / 2) * mass * bezier_curve_first_derivative(t) ** 2 + mass * np.dot(
        bezier_curve_function(t), g)


def get_velocity(g: np.ndarray, mass: float, energy, bezier_curve_first_derivative):
    return lambda t: np.sqrt((2 * (energy(t) - mass * np.dot(bezier_curve_first_derivative(t), g))
                              / (mass * (bezier_curve_first_derivative(t) ** 2))))


def get_acceleration(g: np.ndarray, bezier_curve_first_derivative, bezier_curve_second_derivative, velocity):
    return lambda t: - ((np.dot(bezier_curve_first_derivative(t), g)
                         + (velocity(t) ** 2) * np.dot(bezier_curve_first_derivative(t),
                                                       bezier_curve_second_derivative(t)))
                        / (bezier_curve_first_derivative(t) ** 2))


def get_acceleration2(g: np.ndarray, bezier_curve_first_derivative, bezier_curve_second_derivative):
    return lambda t, s_first_derivative: - ((np.dot(bezier_curve_first_derivative(t), g)
                                             + (s_first_derivative ** 2) * np.dot(bezier_curve_first_derivative(t),
                                                                                  bezier_curve_second_derivative(t)))
                                            / (bezier_curve_first_derivative(t) ** 2))
                                            
def get_dynamic_behaviour(samples) -> typing.Tuple[np.ndarray, np.ndarray]:
    max_s = 10
    data = get_curve_data(samples)
    runge_kutta_integration = 0
    ts = []
    ys = []
    for overall_t in np.linspace(start=0, stop=max_s, num=100):
        nth_curve = int(overall_t % len(samples))
        floating_point_part = overall_t % 1

        # runge_kutta_integration = scipy.integrate.RK45(data["accelerations"][nth_curve](floating_point_part),
        #                                                t0=0.0, y0=p0s[nth_curve], t_bound=1.0)
        runge_kutta_integration = scipy.integrate.RK45(data["accelerations"][nth_curve],
                                                       t0=floating_point_part,
                                                       y0=data["beziers_first_derivative"][nth_curve](
                                                           floating_point_part),
                                                       t_bound=100)
        runge_kutta_integration.step()
        ts.append(runge_kutta_integration.t)
        ys.append(runge_kutta_integration.y)
        print("status: ", runge_kutta_integration.status)
        print("t value (time): ", runge_kutta_integration.t)
        print("y value (state): ", runge_kutta_integration.y)
        # position = runge_kutta_integration.step()
        # print("position: ", position)
    print("status: ", runge_kutta_integration.status)
    return np.asarray(ts), np.asarray(ys)
"""

# trying the "numerische umparametrisierung nach bogenlaenge"
'''
def read_csv_ts(ts_csv_path: str) -> np.ndarray:
    
    """
    Reads the csv file and returns a numpy array.
    :param ts_csv_path: The path to the csv file.
    :return: The ts as np array.
    """
    
    ts = []
    with open(ts_csv_path) as ts_csv:
        csv_reader = csv.reader(ts_csv, delimiter=",")
        for t in csv_reader:
            ts.append(np.array(t, dtype=float))
    return np.asarray(ts)

def save_file(sample_array: np.ndarray, name: str) -> None:
    """
    Saves the given sample array in a file.
    :param sample_array: all samples
    :param name: the name of the file with format ending
    """
    np.savetxt(name, sample_array, delimiter=",")

def numerische_umparametrisierung_nach_bogenlaenge(data):
    print("Starting numeric approach... ")
    n = len(data["beziers"])
    curve_length_separations = n * 200
    t_steps = curve_length_separations * 300
    curve_lengths_at_t = {}
    total_length_at_t = 0
    last_point = data["beziers"][0](0)
    points = []
    ts = []
    t_index = 0
    # In uniform t_steps steps, from 0 to n (0 to 1 for each sample), save curve positions as points and the lengths
    # of the curves up to point t. Save the ts and the t's indexes too.
    print(f"Go through {t_steps} ts... ")
    for t in np.linspace(start=0, stop=n, num=t_steps):
        nth_curve = int(t % n)
        floating_point_part = t % 1
        point = data["beziers"][nth_curve](floating_point_part)
        points.append(point)
        length = abs(float(np.linalg.norm(last_point - point)))
        last_point = point
        total_length_at_t += length
        curve_lengths_at_t[t_index] = total_length_at_t
        ts.append(t)
        t_index += 1
    data["points"] = points

    i_list = []
    total_length = max(curve_lengths_at_t.values())
    length_at_i = []
    # In uniform "cure_length_separations" steps, from 0 to n (0 to 1 for each sample), save percentages of
    # "current" curve lengths from the total length. Also save i, the "helper parameter".
    print(f"Go through {curve_length_separations} i separations... ")
    for i1 in np.linspace(start=0, stop=n, num=curve_length_separations):
        curve_length_in_i_percentage = i1 / n * total_length
        length_at_i.append(curve_length_in_i_percentage)
        i_list.append(i1)

    matching_t_for_i = {}
    # For each curve length at i (uniform helper parameter), find the nearest curve length at a t and
    # save this t's index and the corresponding i
    print("Get the closest curve length and corresponding t for each i... ")
    for index, i2 in enumerate(i_list):
        current_length = length_at_i[index]
        absolute_difference_function = lambda curve_length_at_a_t: abs(curve_length_at_a_t - current_length)
        closest_value = min(curve_lengths_at_t.values(), key=absolute_difference_function)
        matching_t_for_i[i2] = list(curve_lengths_at_t.keys())[list(curve_lengths_at_t.values()).index(closest_value)]
        """
        for t, curve_length in enumerate(curve_lengths_at_t):
            if current_length > curve_length:
                if np.round(curve_length, rounding_floating_nr) == np.round(current_length, rounding_floating_nr):
                    matching_t_for_i[i2] = t
                    break
                # print(curve_length, current_length)
            else:
                break        
        """

    print(matching_t_for_i)
    print(len(matching_t_for_i))
    #### rounding nr: 5, curve separations: 600, ts: 1000 * curve separations, 524 matches, ~ 15min runtime
    new_ts = []
    new_is = []
    curve_positions = []
    # For each i and t-index, find and save the corresponding t and the corresponding position on the curve.
    print("Get the new ts and curve positions 'nach Bogenlaenge parametrisiert'... ")
    for i, t_index in matching_t_for_i.items():
        t = ts[t_index]
        new_is.append(i)
        nth_curve = int(t % n)
        floating_point_part = t % 1
        new_ts.append(t)
        curve_positions.append(data["beziers"][nth_curve](floating_point_part))
    print(new_ts)
    save_file(np.array(new_ts), "parametrized_ts.csv")
    save_file(np.array(new_is), "parametrized_is.csv")
    data["parametrized_ts_bogenlaenge"] = new_ts
    data["parametrized_positions1"] = curve_positions
    return data
'''
