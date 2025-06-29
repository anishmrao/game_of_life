from common import runGOL, TimingDecorator

def count_neighbors(grid, row, col):
    rows = len(grid)
    cols = len(grid[0])
    alive_neighbors = 0

    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if (r == row and c == col):
                continue
            if (0 <= r < rows) and (0 <= c < cols):
                alive_neighbors += grid[r][c]
    return alive_neighbors

@TimingDecorator()  # Uses the static decorator
def update_grid(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            neighbors = count_neighbors(grid, r, c)
            state = grid[r][c]
            # Apply Conway's rules
            if state == 1:
                if neighbors < 2 or neighbors > 3:
                    new_grid[r][c] = 0
                else:
                    new_grid[r][c] = 1
            else:
                if neighbors == 3:
                    new_grid[r][c] = 1
    return new_grid


if __name__ == "__main__":
    runGOL(update_grid)
