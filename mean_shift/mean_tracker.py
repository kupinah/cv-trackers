import numpy as np

from utils import (
    Tracker,
    create_epanechnik_kernel,
    custom_get_patch,
    extract_histogram,
    get_patch,
)
from mean_shift import mean_shift


class MeanTracker(Tracker):
    def initialize(self, image: np.ndarray, region: list) -> None:
        """
        Initialize the tracker with the initial region

        Args:
        ----------
            image: np.ndarray
                The image to track
            region: list
                The initial region to track

        Returns:
        ----------
            None
        """
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [
                np.min(x_),
                np.min(y_),
                np.max(x_) - np.min(x_) + 1,
                np.max(y_) - np.min(y_) + 1,
            ]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        self.kernel = create_epanechnik_kernel(region[2], region[3], self.parameters.sigma)
        self.size = np.shape(self.kernel)
        self.patch, _ = get_patch(image, self.position, (self.size[1], self.size[0]))

        self.q = extract_histogram(self.patch, self.parameters.n_bin, self.kernel)

    def track(self, image: np.ndarray) -> list:
        """
        Track the object in the image

        Args:
        ----------
            image: np.ndarray
                The image to track the object in

        Returns:
        ----------
            list: The bounding box of the tracked object
        """
        
        y, x, self.patch = mean_shift(
            image,
            self.position,
            self.size,
            hist=self.q,
            kernel=self.kernel,
            n_bin=self.parameters.n_bin,
        )

        self.position = (x, y)

        x, y, bbox = custom_get_patch(x, y, self.size[0], self.size[1], image)

        q_new = extract_histogram(bbox, self.parameters.n_bin, self.kernel)

        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * q_new

        return [x, y, self.size[1], self.size[0]]


class MTParams:
    def __init__(self, sigma, n_bin, alpha):
        self.sigma = sigma
        self.n_bin = n_bin
        self.alpha = alpha
