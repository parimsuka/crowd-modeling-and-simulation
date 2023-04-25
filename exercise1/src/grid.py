import pygame
import pygame.gfxdraw

from constants import EMPTY_CELL_COLOR, PEDESTRIAN_COLOR, OBSTACLE_COLOR, TARGET_COLOR, TRACE_COLOR
from utils import draw_rounded_rect
from pedestrian import Pedestrian


class Grid:
    def __init__(self, grid_size, cell_size):
        """
        Initialize the Grid.

        :param grid_size: The number of rows and columns of the grid
        :param cell_size: The size (width and height) of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = [['E' for _ in range(grid_size)] for _ in range(grid_size)]
        self.target_positions = []
        self.pedestrians = []

    def add_pedestrian(self, pedestrian):
        """
        Add a pedestrian at a specific position on the grid.

        :param pedestrian: A pedestrian object, containing the x and y coordinate of the pedestrian and the speed.
        """
        self.grid[int(pedestrian.x)][int(pedestrian.y)] = 'P'
        self.pedestrians.append(pedestrian)

    def add_target(self, x, y):
        """
        Add a target at a specific position on the grid.

        :param x: The x-coordinate of the target's position
        :param y: The y-coordinate of the target's position
        """
        self.grid[x][y] = 'T'
        self.target_positions.append((x, y))

    def add_obstacle(self, x, y):
        """
        Add an obstacle at a specific position on the grid.

        :param x: The x-coordinate of the obstacle's position
        :param y: The y-coordinate of the obstacle's position
        """
        self.grid[x][y] = 'O'

    def get_cell_type(self, x, y):
        return self.grid[x][y]

    def draw(self, win):
        """
        Draw the grid and its elements on the window.

        :param win: The pygame window to draw on
        """
        corner_radius = 1
        for j, row in enumerate(self.grid):  # TODO Tali: Check that swapping i and j as a fix for the x/y reversion does not break anything else
            for i, cell in enumerate(row):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if cell == 'E':
                    color = EMPTY_CELL_COLOR
                elif cell == 'P':
                    # Draw the pedestrian as a circle
                    pygame.draw.ellipse(win, PEDESTRIAN_COLOR, rect)
                    pygame.draw.ellipse(win, (0, 0, 0), rect, 1)
                    continue
                elif cell == 'O':
                    color = OBSTACLE_COLOR
                elif cell == 'T':
                    color = TARGET_COLOR
                elif cell == 'R':
                    color = TRACE_COLOR

                draw_rounded_rect(win, color, rect, corner_radius)
                pygame.draw.rect(win, (0, 0, 0), rect, 1, border_radius=corner_radius)

    def update(self):
        """
        Update the grid by moving pedestrians towards the target.
        """
        # Create a new empty grid with the same size as the current grid
        new_grid = [['E' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        pedestrian_positions = []

        # Move pedestrians
        for ped in self.pedestrians:
            # save old pedestrian position
            old_x, old_y = ped.get_position()
            # calculate best pedestrian move and update its internal position
            new_x, new_y = ped.move_to_closest_target(self.target_positions, self.grid)

            # Handle the new pedestrian position
            # If the new position is a target, the pedestrian is absorbed.
            # (Only absorbing targets are currently implemented)

            # If new position is not target:
            if self.grid[int(new_x)][int(new_y)] != 'T':
                # Add pedestrian to new grid
                new_grid[int(new_x)][int(new_y)] = 'P'
                # Keep track of the new pedestrian position
                pedestrian_positions.append((int(new_x), int(new_y)))

            # If the previous position of the pedestrian is not already taken by another pedestrian,
            # update the old position with a trace
            if (int(old_x), int(old_y)) not in pedestrian_positions:
                new_grid[int(old_x)][int(old_y)] = 'R'
                # Change the previously occupied space on the current (not new) grid to empty, so other pedestrians
                # can go there now
                self.grid[int(old_x)][int(old_y)] = 'E'


        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                # if cell == 'P':
                    # move_i, move_j = self.find_best_move(i, j, self.target_positions)
                    # # If the cell where the pedestrian wants to move is not occupied by the target
                    # if self.grid[move_i][move_j] != 'T':
                    #     new_grid[move_i][move_j] = 'P'
                    #     # Keep track of the new pedestrian position
                    #     pedestrian_positions.append((move_i, move_j))
                    #     # If the previous position of the pedestrian is not already taken by another pedestrian,
                    #     # update the old position with a trace
                    #     if (i, j) not in pedestrian_positions:
                    #         new_grid[i][j] = 'R'
                    # # If the cell where the pedestrian wants to move is occupied by the target
                    # elif self.grid[move_i][move_j] == 'T':
                    #     # If the previous position of the pedestrian is not already taken by another pedestrian,
                    #     # update the old position with a trace
                    #     if (i, j) not in pedestrian_positions:
                    #         new_grid[i][j] = 'R'
                # If the cell contains an obstacle, keep the obstacle in the new grid
                if cell == 'O':
                    new_grid[i][j] = 'O'
                # If the cell contains a target, keep the target in the new grid
                elif cell == 'T':
                    new_grid[i][j] = 'T'
                # If the cell contains a trace and the trace is not occupied by a pedestrian, keep trace in the new grid
                elif cell == 'R' and (i, j) not in pedestrian_positions:
                    new_grid[i][j] = 'R'

        self.grid = new_grid

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

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_i, new_j = i + dx, j + dy
            # check if the new position would be inside the grid
            if 0 <= new_i < self.grid_size and 0 <= new_j < self.grid_size:
                if self.grid[new_i][new_j] not in ('O', 'P'):
                    min_tgt_distance = min(
                        (new_i - tgt_i) ** 2 + (new_j - tgt_j) ** 2 for tgt_i, tgt_j in target_positions)
                    if min_tgt_distance < min_distance:
                        min_distance = min_tgt_distance
                        best_move = (new_i, new_j)

        return best_move
