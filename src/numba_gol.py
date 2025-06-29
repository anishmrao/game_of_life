import numpy as np
from numba import njit, prange
from common import runGOL, TimingDecorator

@TimingDecorator()
@njit(parallel=True, fastmath=True)
def update_grid(grid):
    """
    Compute one Life step using a JIT‑compiled loop.
    * grid : 2‑D uint8 array with values 0 (dead) or 1 (alive).
    Returns a new uint8 array with the next generation.
    """
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)

    for r in prange(rows):                 # prange => multithreaded
        for c in range(cols):
            # Sum eight neighbours with explicit bounds checks
            total = 0
            for dr in (-1, 0, 1):
                nr = r + dr
                if nr < 0 or nr >= rows:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue           # skip self
                    nc = c + dc
                    if nc < 0 or nc >= cols:
                        continue
                    total += grid[nr, nc]

            if grid[r, c]:                 # cell is alive
                new_grid[r, c] = 1 if (total == 2 or total == 3) else 0
            else:                          # cell is dead
                new_grid[r, c] = 1 if total == 3 else 0

    return new_grid

if __name__ == "__main__":
    runGOL(update_grid)
