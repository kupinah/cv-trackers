import matplotlib.pyplot as plt
import numpy as np
from utils import (
    backproject_histogram,
    custom_get_patch,
    extract_histogram,
    generate_responses_1,
)

plt.rc("text", usetex=True)  # use latex for text

def create_kernel_mask(size: tuple) -> tuple:
    """
    Create a kernel mask

    Args:
    ----------
        size: tuple
            Size of the kernel

    Returns:
    ----------
        tuple: Tuple of kernel masks
    """
    h, w = size
    kernel_y = np.tile(np.arange(-h // 2 + 1, h // 2 + 1), w).reshape((w, h)).T
    kernel_x = np.tile(np.arange(-w // 2 + 1, w // 2 + 1), h).reshape((h, w))

    return kernel_x, kernel_y


def mean_shift(
    img: np.ndarray,
    init_pos: tuple,
    kernel_shape: tuple,
    max_iter: int=20,
    tol: int=1,
    hist: np.ndarray=None,
    kernel: np.ndarray=None,
    n_bin: int=None,
    trajectory_color: str=None,
 ) -> tuple:
    """
    Perform mean shift tracking

    Args:
    ----------
        img: np.ndarray
            Image to track
        init_pos: tuple
            Initial position of the object
        kernel_shape: tuple
            Shape of the kernel
        max_iter: int
            Maximum number of iterations
        tol: int
            Tolerance
        hist: np.ndarray
            Histogram
        kernel: np.ndarray
            Kernel
        n_bin: int
            Number of bins
        trajectory_color: str
            Color of the trajectory

    Returns:
    ----------
        tuple: The new position of the object
    """

    x, y = init_pos

    kernel_h, kernel_w = kernel_shape

    x_coord, y_coord = create_kernel_mask(kernel_shape)

    g_x = np.ones_like(x_coord).astype(np.float64)
    g_y = np.ones_like(y_coord).astype(np.float64)

    if trajectory_color is not None:
        trajectory = [(x, y)]

    for _ in range(max_iter):
        if kernel is not None:
            x, y, patch = custom_get_patch(x, y, kernel_h, kernel_w, img)
            p = extract_histogram(patch, n_bin, kernel)
            p /= np.sum(p)
            v = np.sqrt(np.divide(hist, p + 1e-3))
            weights = backproject_histogram(patch, v, n_bin)

        x_new = np.sum(x_coord * weights * g_x) / (np.sum(weights * g_x) + 1e-7)
        y_new = np.sum(y_coord * weights * g_y) / (np.sum(weights * g_y) + 1e-7)

        if np.abs(x_new) < tol and np.abs(y_new) < tol:
            break

        x, y = x + x_new, y + y_new

        if trajectory_color:
            trajectory.append((x, y))

    if trajectory_color:
        plt.plot(
            [x for x, _ in trajectory],
            [y for _, y in trajectory],
            c=trajectory_color,
            marker="x",
            markersize=5,
        )

    return y, x, patch


def main():
    resp = generate_responses_1()
    pos = (50, 50)
    plt.scatter([50], [50], c="black", marker="x", s=100)
    y, x, _ = mean_shift(resp, pos, (5, 5), max_iter=1000, tol=1e-2)

    plt.imshow(resp, cmap="coolwarm")
    plt.scatter([x], [y], c="w", marker="x", s=100)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
