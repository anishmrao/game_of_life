import pygame
import numpy as np
import time
class GameDisplay:
    """Minimal Pygame display for Conway's Game of Life."""
    
    def __init__(self, width: int, height: int, cell_size: int, caption: str = "Game of Life"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.cell_size = cell_size
        self.rows = width // cell_size
        self.cols = height // cell_size

        # Performance tracking
        self.clock = pygame.time.Clock()
        self.last_fps_update = time.time()
        self.frame_count = 0
    
    def draw_grid(self, grid: np.ndarray) -> None:
        """Draw the grid using surfarray for faster rendering."""
        # Create a 3D array (height, width, 3) for RGB values
        height, width = grid.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Set white (255, 255, 255) for live cells, black (0, 0, 0) for dead
        rgb_array[grid == 1] = (255, 255, 255)
        
        # Scale up the array to match cell size if needed
        if self.cell_size > 1:
            rgb_array = np.repeat(np.repeat(rgb_array, self.cell_size, axis=0), 
                               self.cell_size, axis=1)
        
        # Create a surface from the array and blit it
        surf = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
    
    def handle_events(self) -> bool:
        """Handle quit events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True
    
    def tick(self, fps: int) -> None:
        """Control frame rate."""
        self.clock.tick(fps)

        # Update FPS counter every second
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            actual_fps = self.frame_count / (current_time - self.last_fps_update)
            pygame.display.set_caption(f"Game of Life - {actual_fps:.1f} FPS")
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def cleanup(self) -> None:
        """Clean up."""
        pygame.quit()
    
    @property
    def grid_dimensions(self) -> tuple:
        """Get grid dimensions."""
        return (self.rows, self.cols)