import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to draw grid lines
def draw_grid(ax, lines, line_color="blue", alpha=1.0):
    for start, end in lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], line_color, alpha=alpha, zorder=1)

# Function to generate grid lines
def generate_grid(num_lines=10):
    grid_lines = []
    for x in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
        for y in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
            grid_lines.append((np.array([x, -num_lines]), np.array([x, num_lines])))
            grid_lines.append((np.array([-num_lines, x]), np.array([num_lines, x])))
    return grid_lines

# Function to apply matrix transformation on grid lines
def transform_grid(matrix, grid_lines):
    return [(matrix @ start, matrix @ end) for start, end in grid_lines]

# Sidebar
st.sidebar.title("Matrix-Based Linear Transformation")
matrix = np.zeros((2, 2))
matrix[0, 0] = st.sidebar.number_input("Enter matrix element (0, 0)", value=1.0)
matrix[0, 1] = st.sidebar.number_input("Enter matrix element (0, 1)", value=0.0)
matrix[1, 0] = st.sidebar.number_input("Enter matrix element (1, 0)", value=0.0)
matrix[1, 1] = st.sidebar.number_input("Enter matrix element (1, 1)", value=1.0)

show_vector = st.sidebar.checkbox("Show vector")
vector = np.zeros(2)
if show_vector:

    vector[0] = st.sidebar.number_input("Enter vector x-coordinate", value=0.0)
    vector[1] = st.sidebar.number_input("Enter vector y-coordinate", value=0.0)

show_unit_vectors = st.sidebar.checkbox("Show unit vectors")

transform_button = st.sidebar.button("Transform")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Generate the original grid lines
grid_lines = generate_grid()

# Draw the original grid lines
draw_grid(ax, grid_lines, line_color="lightblue", alpha=1.0)
transformed_vector = np.zeros(2)
if transform_button:
    # Apply the transformation to the grid lines
    transformed_lines = transform_grid(matrix, grid_lines)

    # Draw the transformed grid lines
    draw_grid(ax, transformed_lines, line_color="lightcoral", alpha=1.0)

    # Draw the origin as a big dot
    ax.plot(0, 0, marker='o', markersize=10, color="black", zorder=3)

    if show_vector:
        # Transform the input vector
        transformed_vector = matrix @ vector

        # Draw the input vector in the original coordinate space
        ax.arrow(0, 0, vector[0], vector[1], head_width=0.6, head_length=0.9, fc='green', ec='green', linewidth=4.0, zorder=2)

        # Draw the input vector in the transformed coordinate space
        ax.arrow(0, 0, transformed_vector[0], transformed_vector[1], head_width=0.6, head_length=0.9, fc='purple', ec='purple', linewidth=4.0, zorder=2)

    # Print the coordinates of the original and transformed vectors
    ax.text(vector[0], vector[1], f"({vector[0]:.2f}, {vector[1]:.2f})", fontsize=14, color='black', verticalalignment='bottom', horizontalalignment='right')
    ax.text(transformed_vector[0], transformed_vector[1], f"({transformed_vector[0]:.2f}, {transformed_vector[1]:.2f})", fontsize=14, color='black', verticalalignment='bottom', horizontalalignment='right')

    if show_unit_vectors:
        # Transform the unit vectors
        transformed_i = matrix @ np.array([1, 0])
        transformed_j = matrix @ np.array([0, 1])

        # Draw the original unit vectors
        ax.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=4.0, zorder=2)
        ax.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=4.0, zorder=2)

        # Draw the transformed unit vectors
        ax.arrow(0, 0, transformed_i[0], transformed_i[1], head_width=0.2, head_length=0.3, fc='brown', ec='brown', linewidth=4.0, zorder=2)
        ax.arrow(0, 0, transformed_j[0], transformed_j[1], head_width=0.2, head_length=0.3, fc='brown', ec='brown', linewidth=4.0, zorder=2)

        # Add labels for unit vectors
        ax.text(1, 0, "i", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')
        ax.text(0, 1, "j", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')
        ax.text(transformed_i[0], transformed_i[1], "i'", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')
        ax.text(transformed_j[0], transformed_j[1], "j'", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')

matrix_text = f"Matrix:\n[[{matrix[0, 0]:.2f}, {matrix[0, 1]:.2f}]\n[{matrix[1, 0]:.2f}, {matrix[1, 1]:.2f}]]"
plt.text(0.05, 0.95, matrix_text, fontsize=16, transform=ax.transAxes, verticalalignment='top')

st.pyplot(fig)

