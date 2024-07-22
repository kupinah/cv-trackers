import matplotlib.pyplot as plt
import numpy as np

from ex1_utils import rotate_image, show_flow
from of_methods import horn_schunck, lucaskanade

plt.rc("text", usetex=True)  # use latex for text

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)

U_lk_noopt, V_lk_noopt = lucaskanade(im1, im2, 3, False, False)
U_lk_opt, V_lk_opt = lucaskanade(im1, im2, 3)
U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)

fig, ((ax1_11, ax1_12, ax1_13), (ax1_21, ax1_22, ax1_23)) = plt.subplots(2, 3)

fig.text(
    0.176,
    0.99,
    f"Lucas-Kanade\n($N=3$, $\\sigma=1$)",
    ha="center",
    va="top",
    fontsize=10,
)
fig.text(
    0.5,
    0.99,
    f"Lucas-Kanade\n($N=3$, $\\sigma=1$, averaged)",
    ha="center",
    va="top",
    fontsize=10,
)
fig.text(
    0.83,
    0.99,
    f"Horn-Schunck\n($\\alpha=0.5$, $N=1000$)",
    ha="center",
    va="top",
    fontsize=10,
)

# fig.text(0.5, 0.5, "Lucas-Kanade", fontsize=12, ha="center")
show_flow(U_lk_noopt, V_lk_noopt, ax1_11, type="angle")
show_flow(U_lk_noopt, V_lk_noopt, ax1_21, type="field", set_aspect=True)
show_flow(U_lk_opt, V_lk_opt, ax1_12, type="angle")
show_flow(U_lk_opt, V_lk_opt, ax1_22, type="field", set_aspect=True)
show_flow(U_hs, V_hs, ax1_13, type="angle")
show_flow(U_hs, V_hs, ax1_23, type="field", set_aspect=True)

for i in [ax1_11, ax1_12, ax1_21, ax1_22, ax1_13, ax1_23]:
    plt.setp(i.get_xticklabels(), visible=False)
    plt.setp(i.get_yticklabels(), visible=False)
    i.tick_params(axis="both", which="both", length=0)

plt.tight_layout()
plt.show()
