import numpy as np
from common import runGOL, TimingDecorator
from numba import cuda

# ─── CUDA UPDATE KERNEL ───────────────────────────────────────────────────
@cuda.jit
def update_grid_cuda_kernel(grid_d, new_d, rows, cols):
    r, c = cuda.grid(2)
    if r < rows and c < cols:
        total = 0
        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= rows: continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                nc = c + dc
                if nc < 0 or nc >= cols: continue
                total += grid_d[nr, nc]
        if grid_d[r, c]:
            new_d[r, c] = 1 if (total == 2 or total == 3) else 0
        else:
            new_d[r, c] = 1 if (total == 3) else 0

def make_cuda_updater():
    # Allocate device arrays once
    d_grid = None
    d_new = None
    
    # Calculate grid and block dimensions once
    threadsperblock = (16, 16)
    blockspergrid = None

    @TimingDecorator()
    def update_grid(grid):
        nonlocal d_grid, d_new  # Reference the outer function's variables
        nonlocal blockspergrid, threadsperblock
        
        # Copy input to device
        if d_grid is None:
            d_grid = cuda.to_device(grid)
            d_new = cuda.device_array_like(grid)
            blockspergrid = (
                (grid.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
                (grid.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
            )
        else:
            d_grid.copy_to_device(grid)
        
        # Launch kernel
        update_grid_cuda_kernel[blockspergrid, threadsperblock](
            d_grid, d_new, grid.shape[0], grid.shape[1]
        )
        cuda.synchronize()
        
        # Swap device arrays
        d_grid, d_new = d_new, d_grid
        
        # Copy result back to host
        d_grid.copy_to_host(grid)
        return grid
    
    return update_grid

if __name__ == "__main__":
    # Create the update function with the initial grid allocation
    update_grid = make_cuda_updater()
    runGOL(update_grid)
    