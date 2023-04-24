# Constants for the crowd simulation
CELL_SIZE = 20  # The size of each cell in the grid (in pixels)
GRID_SIZE = 25  # The number of cells in each row and column of the grid
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, (GRID_SIZE * CELL_SIZE) + 60  # The dimensions of the window
BACKGROUND_COLOR = (220, 220, 220)  # The background color of the window
EMPTY_CELL_COLOR = (255, 255, 255)  # The color of an empty cell
PEDESTRIAN_COLOR = (30, 144, 255)  # The color of a pedestrian
OBSTACLE_COLOR = (0, 128, 0)  # The color of an obstacle
TARGET_COLOR = (255, 0, 0)  # The color of a target
TRACE_COLOR = (135, 206, 250)  # The color of a trace left by a pedestrian
