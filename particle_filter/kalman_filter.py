import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ex4_utils import kalman_step

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "w"
plt.rcParams["grid.linestyle"] = "dashed"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.color"] = "gray"

np.random.seed(1)


class KalmanFilter:
    def set_up(self) -> None:
        raise NotImplementedError

    def eval_matrices(self, r, q, Q):
        qi = sp.symbols("qi")

        Q = np.array(Q.subs(qi, q), dtype="float")
        R = r * np.identity(2)

        return Q, R

    def kalman_filter(self, A, C, Q, R, x, y):
        sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
        sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

        sx[0] = x[0]
        sy[0] = y[0]
        state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
        state[0] = x[0]
        state[1] = y[0]
        covariance = np.eye(A.shape[0], dtype=np.float32)

        for j in range(1, x.size):
            state, covariance, _, _ = kalman_step(
                A,
                C,
                Q,
                R,
                np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                np.reshape(state, (-1, 1)),
                covariance,
            )
            sx[j] = state[0]
            sy[j] = state[1]

        return sx, sy


class RWKalmanFilter(KalmanFilter):
    def __init__(self):
        self.name = "RW"

    def set_up(self) -> None:
        T, qi = sp.symbols("T qi")
        F = sp.Matrix(np.zeros((2, 2)))
        L = sp.Matrix(np.identity(2))
        H = np.identity(2)

        A = np.array(sp.exp(F * T).subs(T, 1), dtype="float")
        Q = sp.integrate((A * L) * qi * (A * L).T, (T, 0, T)).subs(T, 1)

        return A, Q, H


class NCVKalmanFilter(KalmanFilter):
    def __init__(self):
        self.name = "NCV"

    def set_up(self) -> None:
        T, qi = sp.symbols("T qi")
        F = sp.Matrix(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]))
        L = sp.Matrix(np.array([[0, 0], [0, 0], [1, 0], [0, 1]]))
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        A = np.array(sp.exp(F * T).subs(T, 1), dtype="float")
        Q = sp.integrate((A * L) * qi * (A * L).T, (T, 0, T)).subs(T, 1)

        return A, Q, H


class NCAKalmanFilter(KalmanFilter):
    def __init__(self):
        self.name = "NCA"

    def set_up(self) -> None:
        T, qi = sp.symbols("T qi")
        F = sp.Matrix(
            np.array(
                [
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
        )
        L = sp.Matrix(np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]]))
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        A = np.array(sp.exp(F * T).subs(T, 1), dtype="float")
        Q = sp.integrate((A * L) * qi * (A * L).T, (T, 0, T)).subs(T, 1)

        return A, Q, H


def spiral_curve():
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    return x, y


def jagged_circle():
    angles = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)

    # Add jaggedness to the circle
    x += 0.1 * np.random.randn(100)
    y += 0.1 * np.random.randn(100)

    return x, y


def rectangle():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)

    # Concatenate all sides to form the square
    x = np.concatenate(
        (x, np.ones_like(y), np.linspace(1, 0, 10), np.zeros_like(y))
    ) + 0.1 * np.random.randn(40)

    y = np.concatenate(
        (np.zeros_like(y), y, np.ones_like(y), np.linspace(1, 0, 10))
    ) + 0.1 * np.random.randn(40)

    return x, y


def test_kalman():
    # x, y = spiral_curve()
    # x, y = jagged_circle()
    x, y = rectangle()

    params_sc = [[1, 1, 1, 5, 100], [1, 1, 1, 5, 100]]  # Spiral Curve
    params_jc = [[1, 1, 1], [1, 5, 20]]  # Jagged Circle
    params_r = [[1, 1, 1], [1, 20, 100]]  # Rectangle

    RW = RWKalmanFilter()
    NCV = NCVKalmanFilter()
    NCA = NCAKalmanFilter()

    fig, ax = plt.subplots(3, 3, figsize=(4, 4))

    nrow = 0
    ncol = 0

    params = params_r
    for model in [RW, NCV, NCA]:
        ax[nrow, 0].set_ylabel(f"{model.name}", rotation=90, size="large")
        for i in range(len(params[0])):
            A, Q, H = model.set_up()
            r = params[0][i]
            q_val = params[1][i]

            Q, R = model.eval_matrices(r, q_val, Q)

            sx, sy = model.kalman_filter(A, H, Q, R, x, y)

            if ncol == 0 and nrow == 0:
                labels = ["Kalman", "Original"]
            else:
                labels = [None, None]

            ax[nrow, ncol].plot(sx, sy, marker="o", markersize=3, label=labels[0])
            ax[nrow, ncol].plot(x, y, marker="o", markersize=3, label=labels[1])

            if nrow == 0:
                ax[nrow, ncol].set_title(f"r={r}, q={q_val}", size="large")

            if nrow != 2:
                ax[nrow, ncol].set_xticklabels([])

            if ncol != 0:
                ax[nrow, ncol].set_yticklabels([])

            ax[nrow, ncol].spines["bottom"].set_color("black")
            ax[nrow, ncol].spines["left"].set_color("black")

            ncol += 1

        ncol = 0
        nrow += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_kalman()
