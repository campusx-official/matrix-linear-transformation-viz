import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to draw grid lines
def draw_grid(ax, lines, line_color="blue", alpha=1.0):
    for start, end in lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], line_color, alpha=alpha, zorder=1)

# Function to generate grid lines
def generate_grid(num_lines=10):
    grid_lines = []
    for x in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
        for y in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
            for z in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
                grid_lines.append((np.array([x, y, -num_lines]), np.array([x, y, num_lines])))
                grid_lines.append((np.array([x, -num_lines, z]), np.array([x, num_lines, z])))
                grid_lines.append((np.array([-num_lines, y, z]), np.array([num_lines, y, z])))
    return grid_lines

# Function to apply matrix transformation on grid lines
def transform_grid(matrix, grid_lines):
    return [(matrix @ start, matrix @ end) for start, end in grid_lines]

# Sidebar
st.sidebar.title("Matrix-Based Linear Transformation")
matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        matrix[i, j] = st.sidebar.number_input(f"Enter matrix element ({i}, {j})", value=1.0 if i == j else 0.0)

show_vector = st.sidebar.checkbox("Show vector")
vector = np.zeros(3)
if show_vector:
    for i in range(3):
        vector[i] = st.sidebar.number_input(f"Enter vector {['x', 'y', 'z'][i]}-coordinate", value=0.0)

transform_button = st.sidebar.button("Transform")

# Create the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Generate the original grid lines
grid_lines = generate_grid()

# Draw the original grid lines
draw_grid(ax, grid_lines, line_color="lightblue", alpha=1.0)
transformed_vector = np.zeros(3)
if transform_button:
    # Apply the transformation to the grid lines
    transformed_lines = transform_grid(matrix, grid_lines)

    # Draw the transformed grid lines
    draw_grid(ax, transformed_lines, line_color="lightcoral", alpha=1.0)

    if show_vector:
        # Transform the input vector
        transformed_vector = matrix @ vector

        # Draw the input vector in the original coordinate space
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='green', linewidth=1.5, arrow_length_ratio=0.1)

        # Draw the input vector in the transformed coordinate space
        ax.quiver(0, 0, 0, transformed_vector[0], transformed_vector[1], transformed_vector[2], color='purple', linewidth=1.5, arrow_length_ratio=0.1)

matrix_text = f"Matrix:\n[[{matrix[0, 0]:.2f}, {matrix[0, 1]:.2f}, {matrix[0, 2]:.2f}]\n[{matrix[1, 0]:.2f}, {matrix[1, 1]:.2f}, {matrix[1, 2]:.2f}]\n[{matrix[2, 0]:.2f}, {matrix[2, 1]:.2f}, {matrix[2, 2]:.2f}]]"
ax.text2D(0.05, 0.95, matrix_text, fontsize=16, transform=ax.transAxes, verticalalignment='top')

st.pyplot(fig)


