import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ex2_utils import (
    Tracker,
    backproject_histogram,
    create_epanechnik_kernel,
    custom_get_patch,
    extract_histogram,
    get_patch,
)
from ex4_utils import kalman_step, sample_gauss
from kalman_filter import NCAKalmanFilter, NCVKalmanFilter, RWKalmanFilter
from sequence_utils import VOTSequence

np.random.seed(0)


def patcher(x, y, size, image):
    if x + size[0] / 2 >= image.shape[1] or y + size[1] / 2 >= image.shape[0]:
        x = image.shape[1] - size[0] / 2 - 1
        y = image.shape[0] - size[1] / 2 - 1

    top = int(max(y - size[1] / 2, 0))
    bottom = top + size[1]

    left = int(max(x - size[0] / 2, 0))
    right = left + size[0]

    if bottom > image.shape[0]:
        bottom = image.shape[0] - 1
        top = bottom - size[1]
    if right > image.shape[1]:
        right = image.shape[1] - 1
        left = right - size[0]

    return image[top:bottom, left:right]


def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


class ParticleFilter(Tracker):
    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [
                np.min(x_),
                np.min(y_),
                np.max(x_) - np.min(x_) + 1,
                np.max(y_) - np.min(y_) + 1,
            ]

        self.window = max(region[2], region[3])  # Used for NCC

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        self.kernel = create_epanechnik_kernel(region[2], region[3], self.parameters.sigma)
        self.size = np.shape(self.kernel)
        self.patch = patcher(
            self.position[0], self.position[1], (self.size[1], self.size[0]), image
        )

        self.template = self.patch

        self.q = extract_histogram(self.patch, nbins=self.parameters.n_bin, weights=self.kernel)

        noise_coef = self.parameters.q * min(self.size)
        self.A, Q, _ = self.parameters.model.set_up()
        self.Q, _ = self.parameters.model.eval_matrices(1, noise_coef, Q)

        x = [self.position[0], self.position[1]]

        if isinstance(self.parameters.model, NCVKalmanFilter):
            x.extend([0, 0])
        elif isinstance(self.parameters.model, NCAKalmanFilter):
            x.extend([0, 0, 0, 0])

        self.particles = sample_gauss(x, self.Q, self.parameters.n_particles)
        cv2.drawMarker(
            image,
            (int(self.position[0]), int(self.position[1])),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            10,
            1,
        )

        for particle in self.particles:
            cv2.drawMarker(
                image,
                (int(particle[0]), int(particle[1])),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                10,
                1,
            )

        self.weights = np.ones(self.parameters.n_particles)

    def track(self, image):
        cv2.drawMarker(
            image,
            (int(self.position[0]), int(self.position[1])),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            3,
            1,
        )

        for particle in self.particles:
            cv2.drawMarker(
                image,
                (int(particle[0]), int(particle[1])),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                3,
                1,
            )

        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = self.particles[sampled_idxs.flatten(), :]

        noise = sample_gauss(np.zeros(len(self.A)), self.Q, self.parameters.n_particles)

        self.particles = (self.A @ particles_new.T).T + noise

        for i, particle in enumerate(self.particles):
            patch = patcher(particle[0], particle[1], (self.size[1], self.size[0]), image)
            q = extract_histogram(patch, self.parameters.n_bin, weights=self.kernel)
            d_hell = hellinger(q, self.q)
            self.weights[i] = np.exp(-0.5 * d_hell**2 / 0.1**2)

        self.weights /= np.sum(self.weights)
        x, y = self.weights.dot(self.particles)[:2]

        self.position = (x, y)

        self.patch = patcher(x, y, (self.size[1], self.size[0]), image)

        if self.parameters.use_ncc:
            left = max(round(self.position[0] - float(self.window) / 2), 0)
            top = max(round(self.position[1] - float(self.window) / 2), 0)

            right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
            bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

            if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
                return [
                    self.position[0] + self.size[0] / 2,
                    self.position[1] + self.size[1] / 2,
                    self.size[0],
                    self.size[1],
                ]

            matches = cv2.matchTemplate(self.patch, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

            self.position = (
                left + max_loc[0] + float(self.size[0]) / 2,
                top + max_loc[1] + float(self.size[1]) / 2,
            )

            return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]
        else:
            q = extract_histogram(self.patch, self.parameters.n_bin, weights=self.kernel)
            alpha = self.parameters.alpha
            self.q = (1 - alpha) * self.q + alpha * q

        return [x - self.size[1] / 2, y - self.size[0] / 2, self.size[1], self.size[0]]


class PFParams:
    def __init__(self, model, n_particles, q, sigma, n_bin, alpha, use_ncc=False):
        self.model = model()
        self.n_particles = n_particles
        self.q = q
        self.sigma = sigma
        self.n_bin = n_bin
        self.alpha = alpha
        self.use_ncc = use_ncc
