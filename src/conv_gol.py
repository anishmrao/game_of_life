import numpy as np
from scipy.signal import convolve2d
from common import runGOL, TimingDecorator

# Convolution kernel to count neighbors
KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)

@TimingDecorator()
def update_grid(grid):
    """
    Vectorized update using convolution to count neighbors.
    Returns a new grid for the next generation.
    """
    # Count neighbors via 2D convolution
    neighbors = convolve2d(grid, KERNEL, mode='same', boundary='fill', fillvalue=0)

    # Apply Conway's rules: 
    # 1) Dead cells with exactly 3 neighbors become alive
    birth = (grid == 0) & (neighbors == 3)
    # 2) Alive cells with 2 or 3 neighbors stay alive
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))

    # Create new grid by combining birth & survive
    new_grid = np.zeros_like(grid)
    new_grid[birth | survive] = 1
    return new_grid

if __name__ == "__main__":
    runGOL(update_grid)
