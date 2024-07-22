import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)  # use latex for text

from utils import gaussderiv, gausssmooth, show_flow


def warp_image(flow: np.ndarray, cur_img: np.ndarray) -> np.ndarray:
    """
    Warp the current image using the flow

    Args:
    ----------
        flow: np.ndarray
            Optical flow
        cur_img: np.ndarray
            Current image

    Returns:
    ----------
        np.ndarray: Warped image
    """
    h, w = flow.shape[:2]

    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    prev_img = cv2.remap(cur_img, flow, None, cv2.INTER_LINEAR)

    return prev_img


def create_pyramids(im: np.ndarray, levels: int) -> list:
    """
    Create a pyramid of images

    Args:
    ----------
        im: np.ndarray
            Image to create the pyramid from
        levels: int
            Number of levels in the pyramid

    Returns:
        list: List of images in the pyramid
    """
    pyramids = [im]
    for i in range(levels - 1):
        n_x = (pyramids[i].shape[1] + 1) // 2
        n_y = (pyramids[i].shape[0] + 1) // 2

        pyramids.append(cv2.resize(pyramids[i], (n_x, n_y)))
    return pyramids


def overlay_flow(image: np.ndarray, U:np.ndarray, V:np.ndarray, ax: plt.Axes) -> plt.Axes:
    """
    Overlay the flow on top of the image

    Args:
    ----------
        image: np.ndarray
            Image to overlay the flow on
        U: np.ndarray
            Horizontal flow
        V: np.ndarray
            Vertical flow
        ax: matplotlib.axes.Axes
            Axes to plot on
    """
    show_flow(U, V, ax, type="field", set_aspect=True)
    extent = (0, image.shape[1], -image.shape[0], 0)
    ax.imshow(image, alpha=0.8, extent=extent, cmap="gray")

    return ax


def lucaskanade(
    im1: np.ndarray,
    im2: np.ndarray,
    N: int,
    sigma: int = 1,
    average_grad: bool = True,
    use_corners: bool = True,
) -> None:
    """
    Lucas-Kanade optical flow method

    Args:
    ----------
        im1: np.ndarray
            First image
        im2: np.ndarray
            Second image
        N: int
            Window size
        sigma: int
            Sigma for the Gaussian smoothing
        average_grad: bool
            Whether to average the gradients
        use_corners: bool
            Whether to use the corner response
    """

    # Calculate horizontal and vertical gradients
    Ix_1, Iy_1 = gaussderiv(im1, sigma)
    Ix_2, Iy_2 = gaussderiv(im2, sigma)

    # Average the gradients (improvement technique)
    if average_grad:
        Ix = (Ix_1 + Ix_2) / 2
        Iy = (Iy_1 + Iy_2) / 2
    else:
        Ix = Ix_1
        Iy = Iy_1

    # Smooth images
    smoothed_1 = gausssmooth(im1, sigma)
    smoothed_2 = gausssmooth(im2, sigma)

    # Calculate the temporal gradient
    It = smoothed_2 - smoothed_1

    kernel = np.ones((N, N), dtype=np.float32)

    Ix_sq = cv2.filter2D(Ix**2, -1, kernel)
    Iy_sq = cv2.filter2D(Iy**2, -1, kernel)
    Ix_Iy = cv2.filter2D(Ix * Iy, -1, kernel)
    Ix_It = cv2.filter2D(Ix * It, -1, kernel)
    Iy_It = cv2.filter2D(Iy * It, -1, kernel)

    D = Ix_sq * Iy_sq - Ix_Iy**2

    U = -(Iy_sq * Ix_It - Ix_Iy * Iy_It) / (D + 1e-10)
    V = (Ix_Iy * Ix_It - Ix_sq * Iy_It) / (D + 1e-10)

    if average_grad:  # We can't calculate the corner response on averaged images
        Ix, Iy = gaussderiv(im1, sigma)
        Ix_sq = cv2.filter2D(Ix**2, -1, kernel)
        Iy_sq = cv2.filter2D(Iy**2, -1, kernel)
        Ix_Iy = cv2.filter2D(Ix * Iy, -1, kernel)

        D = Ix_sq * Iy_sq - Ix_Iy**2

    if use_corners:
        corner_response = D - 0.05 * (Ix_sq + Iy_sq) ** 2

        corner_threshold = 1e-6

        U[abs(corner_response) < corner_threshold] = 0
        V[abs(corner_response) < corner_threshold] = 0

    return U, V


def lk_pyramids(
    im1: np.ndarray,
    im2: np.ndarray,
    N: int,
    levels: int,
    average_grad: bool = True,
    multi_level: int = 1,
) -> list:
    """
    Pyramidal Lucas-Kanade optical flow method

    Args:
    ----------
        im1: np.ndarray
            First image
        im2: np.ndarray
            Second image
        N: int
            Window size
        levels: int
            Number of levels in the pyramid
        average_grad: bool
            Whether to average the gradients
        multi_level: int
            Number of iterations at each level

    Returns:
    ----------
        list: List of horizontal and vertical flows
    """
    pyramids_1 = create_pyramids(im1, levels)
    pyramids_2 = create_pyramids(im2, levels)

    pyramids_1 = pyramids_1[::-1]
    pyramids_2 = pyramids_2[::-1]

    u, v = lucaskanade(pyramids_1[0], pyramids_2[0], N, average_grad)
    for level in range(1, levels):

        u = cv2.resize(
            u,
            (pyramids_1[level].shape[1], pyramids_1[level].shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        v = cv2.resize(
            v,
            (pyramids_1[level].shape[1], pyramids_1[level].shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        for _ in range(multi_level):  # Running multiple iterations at each level
            flow = np.array([u.copy(), v.copy()]).transpose(1, 2, 0).astype(np.float32)
            warped_img = warp_image(flow, pyramids_2[level])

            residual_u, residual_v = lucaskanade(pyramids_1[level], warped_img, N, average_grad)

            u += residual_u
            v += residual_v

    return u, v


def hornschunck(im1, im2, N=100, lmbd=1, u_lk=None, v_lk=None) -> tuple:
    """
    Horn-Schunck optical flow method

    Args:
    ----------
        im1 : np.ndarray
            First image
        im2 : np.ndarray
            Second image
        N : int
            Number of iterations
        lmbd : int
            Regularization parameter
        u_lk : np.ndarray
            Horizontal flow from Lucas-Kanade
        v_lk : np.ndarray
            Vertical flow from Lucas-Kanade

    Returns:
    ----------
        tuple: Horizontal and vertical flows
    """

    u = np.zeros_like(im1) if u_lk is None else u_lk
    v = np.zeros_like(im1) if v_lk is None else v_lk

    Laplacian_displacement = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)

    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2

    It = gausssmooth(im2, 1) - gausssmooth(im1, 1)

    for i in range(N):
        u_avg = cv2.filter2D(u, -1, Laplacian_displacement) / 4
        v_avg = cv2.filter2D(v, -1, Laplacian_displacement) / 4

        P = Ix * u_avg + Iy * v_avg + It
        D = lmbd + Ix**2 + Iy**2

        u_prev = u
        v_prev = v

        u = u_avg - Ix * P / D
        v = v_avg - Iy * P / D

        flow_mag = np.mean(np.sqrt(u**2 + v**2))

        flow_mag_prev = np.mean(np.sqrt(u_prev**2 + v_prev**2))

        if (abs(flow_mag - flow_mag_prev) / flow_mag) < 1e-3:
            print("Converged at iteration", i)
            break

    return u, v


def main():
    # Example frame        
    frame_1 = cv2.imread("example_data/office_left.png", cv2.IMREAD_GRAYSCALE) / 255
    frame_2 = cv2.imread("example_data/office_right.png", cv2.IMREAD_GRAYSCALE) / 255

    # Regular Lucas-Kanade
    u_nh, v_nh = lucaskanade(frame_1, frame_2, 3, 1, True, False)

    # Optimized Lucas-Kanade
    u, v = lucaskanade(frame_1, frame_2, 3, 1)

    # Pyramidal Lucas-Kanade
    u_pyr, v_pyr = lk_pyramids(frame_1, frame_2, 3, 3, True, 5)

    # Horn-Schunck initialized with the pyramidal Lucas-Kanade
    u_hs_pyr, v_hs_pyr = hornschunck(frame_1, frame_2, 500, 1, u_pyr, v_pyr)

    # Horn-Schunck initialized with the Lucas-Kanade, 1000 iterations
    u_hs_lk, v_hs_lk = hornschunck(frame_1, frame_2, 1000, 1, u_pyr, v_pyr)


if __name__ == "__main__":
    main()
