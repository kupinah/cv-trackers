import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 100)
r = 1 + 0.5 * np.sin(10 * theta)  # Adding jaggedness

x = r * np.cos(theta)
y = r * np.sin(theta)

plt.plot(x, y)
plt.axis("equal")
plt.title("Jagged Circle")
plt.show()

# Define vertices of a square
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

# Add jaggedness
jagged_vertices = vertices + 0.1 * np.random.randn(*vertices.shape)

# Plot jagged square
plt.plot(jagged_vertices[:, 0], jagged_vertices[:, 1])
plt.title("Jagged Square")
plt.axis("equal")
plt.show()