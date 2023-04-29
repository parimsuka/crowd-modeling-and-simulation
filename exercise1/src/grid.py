import pygame

from constants import (
    EMPTY_CELL_COLOR,
    PEDESTRIAN_COLOR,
    OBSTACLE_COLOR,
    TARGET_COLOR,
    TRACE_COLOR,
)

from pedestriancolors import PedestrianColors
from utils import draw_rounded_rect


class Grid:
    """
      A class representing the grid.

      The grid consists of a number of rows and columns, and each cell in the grid
      has a cell_size in pixels. The grid can contain targets, obstacle and pedestrians.
    """
    def __init__(self, grid_height: int, grid_width: int, cell_size: int) -> None:
        """
        Initialize the Grid.

        :param grid_height: The number of rows of the grid
        :param grid_width: The number of columns of the grid
        :param cell_size: The size (width and height) of each cell in pixels
        """
        self.grid_height: int = grid_width
        self.grid_width: int = grid_width
        self.cell_size: int = cell_size
        self.grid: list[list[str]] = [["E" for _ in range(grid_height)] for _ in range(grid_width)]
        self.target_positions: list = []
        self.pedestrians: list = []

    def add_pedestrian(self, pedestrian) -> None:
        """
        Add a pedestrian at a specific position on the grid.

        :param pedestrian: A pedestrian object, containing the x and y coordinate of the pedestrian and the speed.
        """
        self.grid[int(pedestrian.x)][int(pedestrian.y)] = pedestrian.color.name
        self.pedestrians.append(pedestrian)

    def add_target(self, x: int, y: int, absorbable: bool = False) -> None:
        """
        Add a target at a specific position on the grid.

        :param x: The x-coordinate of the target's position
        :param y: The y-coordinate of the target's position
        :param absorbable: If the target should absorb pedestrians
        """
        if absorbable:
            self.grid[x][y] = 'Ta'
        else:
            self.grid[x][y] = 'Tn'
        # self.grid[x][y] = "T"
        self.target_positions.append((x, y))

    def add_obstacle(self, x: int, y: int) -> None:
        """
        Add an obstacle at a specific position on the grid.

        :param x: The x-coordinate of the obstacle's position
        :param y: The y-coordinate of the obstacle's position
        """
        self.grid[x][y] = "O"

    def draw(self, win: pygame.Surface) -> None:
        """
        Draw the grid and its elements on the window.

        :param win: The pygame window to draw on
        """
        corner_radius: int = 1
        for j, row in enumerate(self.grid):
            for i, cell in enumerate(row):
                rect: pygame.Rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if cell == "E":
                    color: tuple = EMPTY_CELL_COLOR
                if cell.startswith("P"):
                    grid_color = PedestrianColors.get_color_by_name(cell)
                    
                    # Draw the pedestrian as a circle
                    pygame.draw.ellipse(win, grid_color.main_color, rect)
                    pygame.draw.ellipse(win, (0, 0, 0), rect, 1)
                    continue
                elif cell == "O":
                    color: tuple = OBSTACLE_COLOR
                elif cell == "T" or cell == 'Ta' or cell == 'Tn':
                    color: tuple = TARGET_COLOR
                elif cell.startswith("R"):
                    grid_color = PedestrianColors.get_color_by_name(cell)
                    color: tuple = grid_color.main_color
                    
                draw_rounded_rect(win, color, rect, corner_radius)
                pygame.draw.rect(win, (0, 0, 0), rect, 1, border_radius=corner_radius)

    def update(self) -> None:
        """
        Update the grid by moving pedestrians towards the target.
        """

        for ped in self.pedestrians:
            # save old pedestrian position
            old_x: float
            old_y: float
            old_x, old_y = ped.get_position()
            # calculate the best pedestrian move and update its internal position
            new_x: float
            new_y: float
            new_x, new_y = ped.move_to_closest_target(self.target_positions, self.grid)
            # Update trace of pedestrian path (removing old pedestrian position)
            self.grid[int(old_x)][int(old_y)] = ped.color.trace_name

            # If new position is absorbable target: remove pedestrian
            if self.grid[int(new_x)][int(new_y)] == 'Ta':
                self.pedestrians.remove(ped)
            else:
                # Otherwise: Move the Pedestrian
                self.grid[int(new_x)][int(new_y)] = ped.color.name

    def find_best_move(self, i: int, j: int, target_positions: list[tuple[int, int]]) -> tuple[int, int]:
        """
        Find the best move for a pedestrian at position (i, j) based on the shortest distance to a target.

        :param i: The x-coordinate of the pedestrian's position
        :param j: The y-coordinate of the pedestrian's position
        :param target_positions: A list of (x, y) tuples representing target positions
        :return: The (x, y) tuple representing the best move for the pedestrian
        """
        min_distance: float = float("inf")
        best_move: tuple[int, int] = (i, j)

        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            new_i: int = i + dx
            new_j: int = j + dy
            # check if the new position would be inside the grid
            if 0 <= new_i < self.grid_height and 0 <= new_j < self.grid_width:
                if self.grid[new_i][new_j] not in ("O", "P"):
                    min_tgt_distance: float = min(
                        (new_i - tgt_i) ** 2 + (new_j - tgt_j) ** 2
                        for tgt_i, tgt_j in target_positions
                    )
                    if min_tgt_distance < min_distance:
                        min_distance = min_tgt_distance
                        best_move: tuple[int, int] = (new_i, new_j)

        return best_move
