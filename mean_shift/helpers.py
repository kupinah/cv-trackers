import matplotlib.pyplot as plt

from mean_shift import mean_shift
from utils import generate_responses_2, generate_responses_3


def plotting_my_fc():
    """
    Plotting the mean shift algorithm
    """
    resp = generate_responses_2()

    ax1 = plt.subplot(131)
    plt.imshow(resp, cmap="coolwarm")

    pos = (50, 50)
    mean_shift(resp, pos, (10, 10), max_iter=1000, tol=1e-2, trajectory_color="r")

    pos = (50, 85)
    mean_shift(resp, pos, (10, 10), max_iter=1000, tol=1e-2, trajectory_color="r")

    pos = (30, 40)
    mean_shift(resp, pos, (10, 10), max_iter=1000, tol=1e-2, trajectory_color="r")

    pos = (75, 25)
    mean_shift(resp, pos, (10, 10), max_iter=1000, tol=1e-2, trajectory_color="r")

    plt.scatter([50, 50, 30, 75], [50, 85, 40, 25], c="w", marker="x", s=100)
    plt.axis("off")
    ax1.set_title("Starting position influence")

    ax2 = plt.subplot(132)
    plt.imshow(resp, cmap="coolwarm")
    pos = (70, 20)
    mean_shift(resp, pos, (10, 10), max_iter=100, tol=1e-2, trajectory_color="r")  # very nice show

    pos = (70, 20)
    mean_shift(resp, pos, (5, 5), max_iter=100, tol=1e-5, trajectory_color="w")

    plt.axis("off")
    ax2.set_title("Kernel size influence")

    ax3 = plt.subplot(133)
    resp = generate_responses_3()
    plt.imshow(resp, cmap="coolwarm")
    pos = (70, 20)
    mean_shift(resp, pos, (10, 10), max_iter=100, tol=1e-5, trajectory_color="r")  # very nice show

    pos = (70, 20)
    mean_shift(resp, pos, (5, 5), max_iter=100, tol=1e-5, trajectory_color="w")

    ax3.set_title("Magnitude size influence")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plotting_my_fc()
