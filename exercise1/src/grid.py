"""
This module contains the class for the Grid.
"""

import pygame

from constants import (
    EMPTY_CELL_COLOR,
    PEDESTRIAN_COLOR,
    OBSTACLE_COLOR,
    TARGET_COLOR,
    TRACE_COLOR,
)
from utils import draw_rounded_rect


class Grid:
    """
      A class representing the grid.

      The grid consists of a number of rows and columns, and each cell in the grid
      has a cell_size in pixels. The grid can contain targets, obstacle and pedestrians.
    """
    def __init__(self, grid_height, grid_width, cell_size):
        """
        Initialize the Grid.

        :param grid_height: The number of rows of the grid
        :param grid_width: The number of columns of the grid
        :param cell_size: The size (width and height) of each cell in pixels
        """
        self.grid_height = grid_width
        self.grid_width = grid_width
        self.cell_size = cell_size
        self.grid = [["E" for _ in range(grid_height)] for _ in range(grid_width)]
        self.target_positions = []
        self.pedestrians = []

    def add_pedestrian(self, pedestrian):
        """
        Add a pedestrian at a specific position on the grid.

        :param pedestrian: A pedestrian object, containing the x and y coordinate of the pedestrian and the speed.
        """
        self.grid[int(pedestrian.x)][int(pedestrian.y)] = "P"
        self.pedestrians.append(pedestrian)

    def add_target(self, x, y, absorbable=False):
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

    def add_obstacle(self, x, y):
        """
        Add an obstacle at a specific position on the grid.

        :param x: The x-coordinate of the obstacle's position
        :param y: The y-coordinate of the obstacle's position
        """
        self.grid[x][y] = "O"

    def draw(self, win):
        """
        Draw the grid and its elements on the window.

        :param win: The pygame window to draw on
        """
        corner_radius = 1
        for j, row in enumerate(self.grid):
            for i, cell in enumerate(row):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if cell == "E":
                    color = EMPTY_CELL_COLOR
                elif cell == "P":
                    # Draw the pedestrian as a circle
                    pygame.draw.ellipse(win, PEDESTRIAN_COLOR, rect)
                    pygame.draw.ellipse(win, (0, 0, 0), rect, 1)
                    continue
                elif cell == "O":
                    color = OBSTACLE_COLOR
                elif cell == "T" or cell == 'Ta' or cell == 'Tn':
                    color = TARGET_COLOR
                elif cell == "R":
                    color = TRACE_COLOR

                draw_rounded_rect(win, color, rect, corner_radius)
                pygame.draw.rect(win, (0, 0, 0), rect, 1, border_radius=corner_radius)

    def update(self):
        """
        Update the grid by moving pedestrians towards the target.
        """

        for ped in self.pedestrians:
            # save old pedestrian position
            old_x, old_y = ped.get_position()
            # calculate best pedestrian move and update its internal position
            new_x, new_y = ped.move_to_closest_target(self.target_positions, self.grid)
            # Update trace of pedestrian path (removing old pedestrian position)
            self.grid[int(old_x)][int(old_y)] = "R"

            # If new position is absorbable target: remove pedestrian
            if self.grid[int(new_x)][int(new_y)] == 'Ta':
                self.pedestrians.remove(ped)
            else:
                # Otherwise: Move the Pedestrian
                self.grid[int(new_x)][int(new_y)] = 'P'

    def find_best_move(self, i, j, target_positions):
        """
        Find the best move for a pedestrian at position (i, j) based on the shortest distance to a target.

        :param i: The x-coordinate of the pedestrian's position
        :param j: The y-coordinate of the pedestrian's position
        :param target_positions: A list of (x, y) tuples representing target positions
        :return: The (x, y) tuple representing the best move for the pedestrian
        """
        min_distance = float("inf")
        best_move = (i, j)

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
            new_i, new_j = i + dx, j + dy
            # check if the new position would be inside the grid
            if 0 <= new_i < self.grid_height and 0 <= new_j < self.grid_width:
                if self.grid[new_i][new_j] not in ("O", "P"):
                    min_tgt_distance = min(
                        (new_i - tgt_i) ** 2 + (new_j - tgt_j) ** 2
                        for tgt_i, tgt_j in target_positions
                    )
                    if min_tgt_distance < min_distance:
                        min_distance = min_tgt_distance
                        best_move = (new_i, new_j)

        return best_move
