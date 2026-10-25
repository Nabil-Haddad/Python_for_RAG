import matplotlib.pyplot as plt
import numpy as np


vector_a = np.array([4, 3])
vector_b = np.array([1, 1])

# Calculate Dot Product
dot_product = np.dot(vector_a, vector_b)

# 3 Calculate Cosine Similarity
norm_a = np.linalg.norm(vector_a)
norm_b = np.linalg.norm(vector_b)
cosine_sim = dot_product / (norm_a * norm_b)

# 4. Plot the vectors
plt.figure(figsize=(7, 7))

# Origins (0, 0) for both arrows
X = [0, 0]
Y = [0, 0]

# Extract directions
U = [vector_a[0], vector_b[0]]
V = [vector_a[1], vector_b[1]]

# Draw arrows from the origin
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=['coral', 'dodgerblue'], zorder=3)

# Annotate the vectors
plt.text(vector_a[0] + 0.1, vector_a[1] + 0.1, "Vector A (4, 3)", fontsize=11, color='coral', weight='bold')
plt.text(vector_b[0] + 0.1, vector_b[1] + 0.1, "Vector B (1, 1)", fontsize=11, color='dodgerblue', weight='bold')

# Add metrics text box to the plot
metrics_text = f"Dot Product: {dot_product:.2f}\nCosine Similarity: {cosine_sim:.4f}"
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Graph styling and boundaries
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)

plt.title("Visualising the Difference: Dot Product vs Cosine Similarity", fontsize=13, pad=15)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

plt.show()