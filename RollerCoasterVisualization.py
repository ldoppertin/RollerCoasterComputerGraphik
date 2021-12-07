"""
PLEASE SPECIFY THE PATH OF THE .csv OR .trk FILE IN THE RUN/DEBUG CONFIGURATIONS TO RUN THIS CODE.
i.e. ENTER e.g. -path TorusSamples_3_7_20.csv OR -path C:/Users/User/Desktop/_WildeMaus2.trk
csv-FILES WITH SAMPLES FROM A TORUS KNOT CAN BE CREATED WITH THE SCRIPT "TorusKnotGeneration.py".
"""

import argparse
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from Modules.CurveGeneration import get_curve_data, read_file
from Modules.MathHelpers import get_geometric_behaviour

parser = argparse.ArgumentParser()
parser.add_argument("-filepath")
args = parser.parse_args()

# this 'keyword' makes the visualization interactable
mpl.use('Qt5Agg')


class RollerCoaster:
    """
    This class simulates a Roller Coaster with the help of bezier-interpolation, but without considering dynamic
    behaviour (no gravity, energy etc.). Additionally, differential geometric measures are displayed within the output
    window.
    """
    Data: typing.Dict
    GeometricValues: typing.Dict
    Ax1: plt.axes
    Ax2: plt.axes
    Ax3: plt.axes
    Ax4: plt.axes
    Ax5: plt.axes
    Ax6: plt.axes
    NextPoint = 0
    NextNormal = 0
    NextBinormal = 0
    NextTangent = 0
    ColorMapper = 0

    def update(self, frame_number) -> None:
        """
        Update-function used by matplotlib's FuncAnimation. Is called at each frame.
        """
        # remove the previously drawn points and vectors
        if self.NextPoint:
            self.NextPoint.remove()
            self.NextNormal.remove()
            self.NextBinormal.remove()
            self.NextTangent.remove()
        # set the new index using the current frame number
        index = frame_number % len(self.Data["points"])

        # update all output values:
        point_coords = self.Data["points"][index]
        # set and draw the colour of the wagon according to the current curvature (de: "Kruemmung")
        curvature_color = self.ColorMapper.to_rgba(self.GeometricValues["curvature_norms"][index])
        self.NextPoint = self.Ax1.scatter3D(point_coords[0], point_coords[1], point_coords[2], marker='o',
                                            color=curvature_color, linewidths=2)
        # set and draw the normal-, binormal- and tangent-vector (de: "Frenet-Dreibein")
        normal = self.GeometricValues["normals"][index]
        self.NextNormal = self.Ax1.quiver(point_coords[0], point_coords[1], point_coords[2], normal[0], normal[1],
                                          normal[2], color='r')
        binormal = self.GeometricValues["binormals"][index]
        self.NextBinormal = self.Ax1.quiver(point_coords[0], point_coords[1], point_coords[2], binormal[0], binormal[1],
                                            binormal[2], color='g')
        tangent = self.GeometricValues["tangents"][index]
        self.NextTangent = self.Ax1.quiver(point_coords[0], point_coords[1], point_coords[2], tangent[0], tangent[1],
                                           tangent[2], color='b')

    def set_color_mapper(self) -> None:
        """
        Set the color mapper used for visualizing the curvature of the Roller Coaster.
        """
        curvature_min = min(self.GeometricValues["curvature_norms"])
        curvature_max = max(self.GeometricValues["curvature_norms"])
        norm = mpl.colors.Normalize(vmin=curvature_min, vmax=curvature_max, clip=True)
        self.ColorMapper = cm.ScalarMappable(norm=norm, cmap=cm.cmaps_listed['viridis'])

    def add_point_calculations_to_data(self, n_points_per_curve: int) -> None:
        """
        Realize the given functions in self.Data by calculating n (time) points per (bezier) curve. These are necessary
        for the animation and calculation of other geometrical measures at precisely these points.
        :param n_points_per_curve: The number of points per distinct curve in the data (to be calculated).
        """
        n_curves = len(self.Data["beziers"])
        self.Data["points"] = []
        self.Data["first_derivative_points"] = []
        self.Data["second_derivative_points"] = []
        self.Data["third_derivative_points"] = []
        self.Data["curve_lengths_at_t"] = []
        total_length = 0
        last_point = self.Data["beziers"][0](0)
        for t in np.linspace(start=0, stop=n_curves, num=n_curves * n_points_per_curve):
            nth_curve = int(t % n_curves)
            floating_point_part = t % 1
            point = self.Data["beziers"][nth_curve](floating_point_part)
            self.Data["points"].append(point)
            first_derivatives_point = self.Data["beziers_first_derivative"][nth_curve](floating_point_part)
            self.Data["first_derivative_points"].append(first_derivatives_point)
            second_derivatives_point = self.Data["beziers_second_derivative"][nth_curve](floating_point_part)
            self.Data["second_derivative_points"].append(second_derivatives_point)
            third_derivatives_point = self.Data["beziers_third_derivative"][nth_curve](floating_point_part)
            self.Data["third_derivative_points"].append(third_derivatives_point)
            length = abs(float(np.linalg.norm(last_point - point)))
            last_point = point
            total_length += length
            self.Data["curve_lengths_at_t"].append(total_length)

    def draw_roller_coaster(self) -> None:
        """
        Read and process the data, then draw the Roller Coaster and corresponding geometrical measures.
        """
        torus_sample_points = SAMPLES
        self.Data = get_curve_data(torus_sample_points)
        self.add_point_calculations_to_data(5)
        self.GeometricValues = get_geometric_behaviour(self.Data["first_derivative_points"],
                                                       self.Data["second_derivative_points"],
                                                       self.Data["third_derivative_points"])
        self.set_color_mapper()
        points = np.array(self.Data["points"])

        # 3d plot of the roller coaster trail
        fig = plt.figure()
        self.Ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.Ax1.set_xlim3d(-5, 5)
        self.Ax1.set_ylim3d(-5, 5)
        self.Ax1.set_zlim3d(-5, 5)
        self.Ax1.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1)

        # plot of the curve length at each time point (added up)
        self.Ax2 = fig.add_subplot(5, 2, 1)
        self.Ax2.set_ylabel("curve length")
        self.Ax2.plot(np.array(self.Data["curve_lengths_at_t"]), linewidth=1)

        # plot of the torsion at each time point (de: "Torsion")
        self.Ax3 = fig.add_subplot(5, 2, 3)
        self.Ax3.set_ylabel("torsion")
        self.Ax3.plot(np.array(self.GeometricValues["torsions"]), linewidth=1)

        # plot of the speeds at each time point (de: "Tempo")
        self.Ax4 = fig.add_subplot(5, 2, 5)
        self.Ax4.set_ylabel("speed")
        self.Ax4.plot(np.array(self.GeometricValues["speeds"]), linewidth=1)

        # plot of the acceleration at each time point (de: "Beschleunigung")
        self.Ax5 = fig.add_subplot(5, 2, 7)
        self.Ax5.set_ylabel("acceleration")
        self.Ax5.plot(np.array(self.GeometricValues["accelerations"]), linewidth=1)

        # plot of the jolt at each time point (de: "Ruck")
        self.Ax6 = fig.add_subplot(5, 2, 9)
        self.Ax6.set_ylabel("jolt")
        self.Ax6.plot(np.array(self.GeometricValues["jolts"]), linewidth=1)
        self.Ax6.legend(["x", "y", "z"], loc='best', bbox_to_anchor=(1.2, 1.2))
        self.Ax6.set_xlabel("time-point")

        # draw the animation (more calculation in the update function)
        animation = FuncAnimation(fig, self.update, interval=100)
        plt.colorbar(mappable=self.ColorMapper, ax=self.Ax1, label="curvature norm")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    SAMPLES = read_file(args.filepath)
    coaster = RollerCoaster()
    coaster.draw_roller_coaster()
    print("Script finished.")
