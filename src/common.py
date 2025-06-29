import time
from statistics import mean
import argparse
import numpy as np
import sys
from display import GameDisplay

class TimingDecorator:
    """A decorator class that measures and stores execution times of a function.
    
    This can be used in multiple ways:
    1. As a decorator: @TimingDecorator
    2. As a context manager: with TimingDecorator() as td:
    """
    _timings = []
    _disabled = False
    
    def __call__(self, func):
        """The decorator function that wraps the original function."""
        def wrapper(*args, **kwargs):
            if self._disabled:
                return func(*args, **kwargs)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            TimingDecorator._timings.append(end_time - start_time)
            return result
        return wrapper
    
    @classmethod
    def average_time(cls) -> float:
        """Return the average execution time of all timed functions."""
        if not cls._timings:
            return 0.0
        return mean(cls._timings)
    
    @classmethod
    def clear_timings(cls):
        """Clear any stored timings so we can start fresh."""
        cls._timings.clear()
    
    @classmethod
    def disable(cls):
        cls._disabled = True
    
    @classmethod
    def enable(cls):
        cls._disabled = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Conway's Game of Life Simulation")
    parser.add_argument('--width', type=int, default=1000,
                       help='Width of the window in pixels (default: 1000)')
    parser.add_argument('--height', type=int, default=1000,
                       help='Height of the window in pixels (default: 1000)')
    parser.add_argument('--cell-size', type=int, default=1,
                       help='Size of each cell in pixels (default: 1)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--show', action='store_true',
                       help='Show the visualization (default: False)')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Number of iterations to run performance measurements (default: 1000)')
    parser.add_argument('--warmup-iterations', type=int, default=50,
                       help='Number of warmup iterations (default: 50)')
    parser.add_argument('--output-file', type=str, default='../analysis/output.csv',
                       help='Output file for performance measurements (default: ../analysis/output.csv)')
    parser.add_argument('--name', type=str, default='Default',
                       help='Name of the experiment (default: Default)')
    return parser.parse_args()

def initialize_grid(rows, cols):
    """Initialize a grid with random values using NumPy for better performance."""
    print(f"Initializing grid with {rows*cols} cells")
    np.random.seed(0)
    return np.random.randint(2, size=(rows, cols), dtype=np.uint8)

def write_to_file(filename, name, num_cells, average_time, num_iterations):
    with open(filename, 'a') as f:
        f.write(f"{name},{num_cells},{average_time},{num_iterations}\n")

def runGOL(update_func):
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize display if show flag is set
    display = None
    if args.show:
        print("Warning: Performance measurements are not enabled when visualization is enabled.")
        TimingDecorator.disable()
        display = GameDisplay(args.width, args.height, args.cell_size, "Conway's Game of Life")
        grid_rows, grid_cols = display.grid_dimensions
    else:
        # Calculate grid dimensions without initializing display
        grid_cols = args.width // args.cell_size
        grid_rows = args.height // args.cell_size
    
    # Initialize grid with random values
    grid = initialize_grid(grid_rows, grid_cols)

    if args.show:
        display.draw_grid(grid)

    running = True
    iteration_count = 0
    
    try:
        while running:                
            # Update the grid
            grid = update_func(grid)
            
            # Draw the grid if display is enabled
            if args.show:
                display.draw_grid(grid)
                running = display.handle_events()
                display.tick(args.fps)
                continue
            
            # Performance measurement
            iteration_count += 1
            if iteration_count < args.warmup_iterations:
                continue
            elif iteration_count == args.warmup_iterations:
                TimingDecorator.clear_timings()
            elif (iteration_count - args.warmup_iterations) == args.max_iterations:
                avg_time = TimingDecorator.average_time()
                write_to_file(args.output_file, args.name, grid_rows * grid_cols, avg_time, args.max_iterations)
                print(f"Average update time per iteration after {args.max_iterations} iterations: {avg_time:.6f} seconds")
                TimingDecorator.clear_timings()
                running = False
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if args.show:
            display.cleanup()
        sys.exit()
