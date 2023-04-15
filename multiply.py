import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def draw_grid(ax, lines, line_color="blue", alpha=1.0):
    for start, end in lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], line_color, alpha=alpha, zorder=1)


def generate_grid(num_lines=10):
    grid_lines = []
    for x in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
        for y in np.linspace(-num_lines, num_lines, 2 * num_lines + 1):
            grid_lines.append((np.array([x, -num_lines]), np.array([x, num_lines])))
            grid_lines.append((np.array([-num_lines, x]), np.array([num_lines, x])))
    return grid_lines


def transform_grid(matrix, grid_lines):
    return [(matrix @ start, matrix @ end) for start, end in grid_lines]


st.title("Matrix-Based Linear Transformation")

row_A1 = st.sidebar.text_input("Enter the first row of matrix A (comma-separated)", value="1,0")
row_A2 = st.sidebar.text_input("Enter the second row of matrix A (comma-separated)", value="0,1")

row_B1 = st.sidebar.text_input("Enter the first row of matrix B (comma-separated)", value="1,0")
row_B2 = st.sidebar.text_input("Enter the second row of matrix B (comma-separated)", value="0,1")

matrix_A = np.array([list(map(float, row_A1.split(','))), list(map(float, row_A2.split(',')))])

matrix_B = np.array([list(map(float, row_B1.split(','))), list(map(float, row_B2.split(',')))])

calculate_button = st.sidebar.button("Calculate")

grid_lines = generate_grid()

if calculate_button:

    transformed_lines_AB = transform_grid(matrix_A @ matrix_B, grid_lines)
    transformed_lines_B = transform_grid(matrix_B, grid_lines)
    transformed_lines_AonB = transform_grid(matrix_A, transformed_lines_B)
    transformed_lines_BA = transform_grid(matrix_B @ matrix_A, grid_lines)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 40), gridspec_kw={'height_ratios': [3, 1.5, 1.5, 3]})

    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    draw_grid(ax1, grid_lines, line_color="lightblue", alpha=1.0)
    draw_grid(ax1, transformed_lines_AB, line_color="lightcoral", alpha=1.0)

    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    draw_grid(ax2, grid_lines, line_color="lightblue", alpha=1.0)
    draw_grid(ax2, transformed_lines_B, line_color="lightcoral", alpha=1.0)

    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    draw_grid(ax3, grid_lines, line_color="lightblue", alpha=1.0)
    draw_grid(ax3, transformed_lines_AonB, line_color="lightcoral", alpha=1.0)

    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-10, 10)
    draw_grid(ax4, grid_lines, line_color="lightblue", alpha=1.0)
    draw_grid(ax4, transformed_lines_BA, line_color="lightcoral", alpha=1.0)

    st.pyplot(fig)

