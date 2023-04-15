import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Geometric Interpretation of Determinants")

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

# Sidebar
st.sidebar.title("Matrix-Based Linear Transformation")
matrix = np.zeros((2, 2))
matrix[0, :] = st.sidebar.text_input("Enter matrix row 0 (comma-separated)", value="1,0").split(',')
matrix[1, :] = st.sidebar.text_input("Enter matrix row 1 (comma-separated)", value="0,1").split(',')
matrix = matrix.astype(np.float64)

transform_button = st.sidebar.button("Transform")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

for ax in [ax1, ax2]:
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    # Generate the original grid lines
    grid_lines = generate_grid()

    # Draw the original grid lines
    draw_grid(ax, grid_lines, line_color="lightblue", alpha=1.0)

if transform_button:
    # Apply the transformation to the unit vectors
    transformed_i = matrix @ np.array([1, 0])
    transformed_j = matrix @ np.array([0, 1])

    # Draw the transformed unit vectors
    ax2.arrow(0, 0, transformed_i[0], transformed_i[1], head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=2.0, zorder=2)
    ax2.arrow(0, 0, transformed_j[0], transformed_j[1], head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=2.0, zorder=2)

    ax2.text(transformed_i[0], transformed_i[1], "i'", fontsize=16, color='black', verticalalignment='bottom',
             horizontalalignment='right')
    ax2.text(transformed_j[0], transformed_j[1], "j'", fontsize=16, color='black', verticalalignment='bottom',
             horizontalalignment='right')

    # Draw the transformed grid lines
    transformed_grid_lines = [(matrix @ start, matrix @ end) for start, end in grid_lines]
    draw_grid(ax2, transformed_grid_lines, line_color="red", alpha=0.5)

    # Draw the transformed square
    transformed_square_points = matrix @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    ax2.fill(transformed_square_points[0], transformed_square_points[1], color='yellow', alpha=0.5)

    # Calculate and display the area of the transformed square
    area = abs(np.linalg.det(matrix))
    ax2.text(sum(transformed_square_points[0]) / 4, sum(transformed_square_points[1]) / 4, f"Area = {area:.2f}", fontsize=14, color='black', verticalalignment='center', horizontalalignment='center')

# Draw the unit vectors on the original plot
ax1.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=2.0, zorder=2)
ax1.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.3, fc='black', ec='black', linestyle="--", linewidth=2.0, zorder=2)
square_points = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
ax1.fill(square_points[0], square_points[1], color='yellow', alpha=0.5)

ax1.text(1, 0, "i", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')
ax1.text(0, 1, "j", fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='right')

st.pyplot(fig)