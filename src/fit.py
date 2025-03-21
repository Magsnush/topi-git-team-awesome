"""Contains the necessary tools to fit a dataset."""

import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# We need the value of the earth gravity to determine the initial velocity and initial angle
from generate import EARTH_GRAVITY

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
PLOT_DIR = os.path.join(REPO_DIR, "plots")

# Step 1: Determine the starting velocity and starting angle of the test dataset `data/A-test.txt`.

## 1. Load data file using https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html#numpy-loadtxt .
##    The first column corresponds to the abscissa x and the second column to the ordinate y.
# See the main section below.


## 2. Define the model: Look to the example on SciPy (the link below)
def model(x, a, b):
    return a * x + b * x**2


## 3. Fit using https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#curve-fit
def fit(x: np.ndarray, y: np.ndarray) -> tp.Tuple[float, float]:
    sol = curve_fit(model, xdata=x, ydata=y)
    a = sol[0][0]
    b = sol[0][1]
    return convert(a, b)


## 4. Calculate the initial velocity and initial angle from the model parameters.
##    (Yes, this is a math exercise ;-) ) Compare your formulae with the Programmer.
def convert(a, b):
    vx = np.sqrt(-EARTH_GRAVITY / (2*b))
    vy = a * vx
    v = np.sqrt(vx**2 + vy**2)
    alpha = np.arctan(vy / vx)
    return v, alpha


# Step 2: Plot the data and your fit into one diagram.

## The most popular library for making plots is matplotlib:
## Homepage: https://matplotlib.org/
## A Quick start example can be found here: https://matplotlib.org/stable/users/explain/quick_start.html
## A list with tutorials can be found here: https://matplotlib.org/stable/tutorials/index.html
## - Save the final plot in the `plots/` directory as `.pdf`
##   The relevant command is described here: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig
## - Note it is often convenient to not commit generated files to repositories, but their recipe instead!
##   (That is why there is a `.gitignore` file in the `plots/` folder.)

def simulate(v: float, alpha: float, t_max: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, t_max, 20)
    vx = v * np.cos(alpha)
    vy = v * np.sin(alpha)
    x = vx * t
    y = vy * t - 0.5 * EARTH_GRAVITY * t**2
    return x, y


def plot(x: np.ndarray, y: np.ndarray, name: str = None) -> plt.Figure:
    v, alpha = fit(x, y)
    x_sim, y_sim = simulate(v, alpha, 4)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y, label="data")
    ax.plot(x_sim, y_sim, label=rf"simulated $v={v:.2f}, \alpha={alpha:.2f}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()

    if name is not None:
        fig_path = os.path.join(PLOT_DIR, name)
        fig.savefig(f"{fig_path}.pdf")
        fig.savefig(f"{fig_path}.svg")
    return fig


if __name__ == "__main__":
    data = np.loadtxt(os.path.join(DATA_DIR, "A-test.txt"))
    fig = plot(data[:, 0], data[:, 1], "Test A")
    plt.show()
