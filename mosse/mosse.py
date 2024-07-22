import cv2
import numpy as np

from utils import (
    create_cosine_window,
    create_gauss_peak,
    custom_get_patch,
)


class MOSSE():
    def name(self):
        return "mosse"

    def __init__(self) -> None:
        self.enlargment_factor = 1.15
        self.sigma = 2
        self.alpha = 0.03
        self.filter = None

    def _get_template(self, gt_bbox):
        self.gt_bbox = gt_bbox

        x, y, w, h = gt_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        _, _, template = custom_get_patch(x, y, h, w, self.img)

        self.x_center = x + w // 2
        self.y_center = y + h // 2

        return template

    def get_filter_h(self, g_hat, f_hat_conj, f_hat, use_improved=True):
        if use_improved:
            if self.filter is None:
                self.nominator = g_hat * f_hat_conj
                self.denominator = f_hat * f_hat_conj + 1000

                self.filter = self.nominator / self.denominator
            else:
                self.nominator = (
                    self.alpha * g_hat * f_hat_conj + (1 - self.alpha) * self.nominator
                )
                self.denominator = (
                    self.alpha * f_hat * f_hat_conj + (1 - self.alpha) * self.denominator
                )

                self.filter = self.nominator / self.denominator
        else:
            if self.filter is None:
                self.filter = g_hat * f_hat_conj / (f_hat * f_hat_conj + 1000)
            else:
                self.filter = (1 - self.alpha) * self.filter + self.alpha * (
                    g_hat * f_hat_conj / (f_hat * f_hat_conj + 1000)
                )

    def initialize(self, image, bbox):
        self.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.img = (self.img - np.mean(self.img)) / np.std(self.img)

        template = self._get_template(bbox)

        h_template, w_template = template.shape
        self.w_template = w_template
        self.h_template = h_template

        h, w = map(int, [h_template * self.enlargment_factor, w_template * self.enlargment_factor])

        if h % 2 == 0:
            h += 1

        if w % 2 == 0:
            w += 1

        self.width = w
        self.height = h

        g = create_gauss_peak((w, h), self.sigma)

        self.gauss = np.fft.fft2(g)
        self.hanning = create_cosine_window((w, h))
        self.search_region, _ = self.get_search_region()

        if self.search_region.shape[0] != self.height or self.search_region.shape[1] != self.width:
            self.width = self.search_region.shape[1]
            self.height = self.search_region.shape[0]

            self.hanning = create_cosine_window((self.width, self.height))

            self.gauss = create_gauss_peak((self.width, self.height), self.sigma)

        self.f_hat = np.fft.fft2(self.search_region * self.hanning)
        self.filter = None
        self.get_filter_h(self.gauss, self.f_hat.conj(), self.f_hat)

    def get_search_region(self):
        x_shifted = self.x_center - self.width // 2
        y_shifted = self.y_center - self.height // 2

        _, _, search_region = custom_get_patch(
            x_shifted, y_shifted, self.height, self.width, self.img
        )

        sr_bbox = (x_shifted, y_shifted, self.width, self.height)

        return search_region, sr_bbox

    def track(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = (self.img - np.mean(self.img)) / np.std(self.img)

        inverse_fft = np.fft.ifft2(self.f_hat * self.filter)

        y, x = np.unravel_index(np.argmax(inverse_fft), inverse_fft.shape)

        if x >= self.width // 2:
            x -= self.width

        if y >= self.height // 2:
            y -= self.height

        self.x_center += x
        self.y_center += y

        bbox = (
            self.x_center - self.w_template // 2,
            self.y_center - self.h_template // 2,
            self.w_template,
            self.h_template,
        )

        self.search_region, _ = self.get_search_region()

        if self.search_region.shape[0] != self.height or self.search_region.shape[1] != self.width:
            self.width = self.search_region.shape[1]
            self.height = self.search_region.shape[0]

            self.hanning = create_cosine_window((self.width, self.height))

            self.gauss = create_gauss_peak((self.width, self.height), self.sigma)

        self.f_hat = np.fft.fft2(self.search_region * self.hanning)

        self.get_filter_h(self.gauss, self.f_hat.conj(), self.f_hat)

        return bbox
