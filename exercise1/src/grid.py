"""
Grid management module for simulation.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


import pygame
from constants import (EMPTY_CELL_COLOR, OBSTACLE_COLOR, PEDESTRIAN_COLOR,
                       TARGET_COLOR, TRACE_COLOR)
from pedestriancolors import PedestrianColors
from visualisation import draw_rounded_rect


class Grid:
    """
    A class representing the grid.

    The grid consists of a number of rows and columns, and each cell in the grid
    has a cell_size in pixels. The grid can contain targets, obstacle and pedestrians.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        cell_size: int,
        measure_start: int,
        measure_stop: tuple[int, int],
    ) -> None:
        """
        Initialize the Grid.

        :param grid_height: The number of rows of the grid
        :param grid_width: The number of columns of the grid
        :param cell_size: The size (width and height) of each cell in pixels
        """
        self.grid_height: int = grid_width
        self.grid_width: int = grid_width
        self.cell_size: int = cell_size
        self.grid: list[list[str]] = [
            ["E" for _ in range(grid_height)] for _ in range(grid_width)
        ]
        self.target_positions: list = []
        self.pedestrians: list = []
        # Point where we want to start the measurement
        self.measure_start_step: int = measure_start
        # Point where we want to stop measuring
        self.measure_stop_step: tuple[int, int] = measure_stop
        # Distance of run by each pedestrian during the measurement
        self.measure_x_positions: list[float] = []
        self.dijkstra_distance = None

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
            self.grid[x][y] = "Ta"
        else:
            self.grid[x][y] = "Tn"
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
                elif cell == "T" or cell == "Ta" or cell == "Tn":
                    color: tuple = TARGET_COLOR
                elif cell.startswith("R"):
                    grid_color = PedestrianColors.get_color_by_name(cell)
                    color: tuple = grid_color.main_color

                draw_rounded_rect(win, color, rect, corner_radius)
                # pygame.draw.rect(win, (0, 0, 0), rect, 1, border_radius=corner_radius)

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
            new_x, new_y = ped.move_to_closest_target(
                self.target_positions, self.grid, self.dijkstra_distance
            )
            # Update trace of pedestrian path (removing old pedestrian position)
            self.grid[int(old_x)][int(old_y)] = ped.color.trace_name

            # If new position is absorbable target: remove pedestrian
            if self.grid[int(new_x)][int(new_y)] == "Ta":
                self.pedestrians.remove(ped)
            else:
                # Otherwise: Move the Pedestrian
                self.grid[int(new_x)][int(new_y)] = ped.color.name

    def find_best_move(
        self, i: int, j: int, target_positions: list[tuple[int, int]]
    ) -> tuple[int, int]:
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

    def dijkstra(self, target_x, target_y):
        """
        Dijkstra algorithm to create the distance map from the target to all other cells in the grid.
        :param target_x: x-coordinate of the target cell
        :param target_y: y-coordinate of the target cell
        """
        # Initialize the distance map
        self.dijkstra_distance = [
            [float("inf") for _ in range(len(self.grid[0]))]
            for _ in range(len(self.grid))
        ]

        # Initialize costs for each cell in the grid
        cost_grid = [
            [1 for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))
        ]
        # Obstacles have infinite cost (we don't want to go through them)

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if self.grid[x][y] == "O":
                    cost_grid[x][y] = 1e10

        # Initialize the visited cells
        visited = [
            [False for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))
        ]
        """
        # Mark obstacles as visited
        for x in range (len(self.grid)):
            for y in range (len(self.grid[0])):
                if self.grid[x][y] == 'O':
                    visited[x][y] = True
        """
        # Begin the search from the target cell
        self.dijkstra_distance[target_x][target_y] = 0
        current_cell = [target_x, target_y]
        while True:
            # Get the neighbors of the current cell
            for i, j in [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1],
            ]:
                neighbor_cell = [current_cell[0] + i, current_cell[1] + j]
                if 0 <= neighbor_cell[0] < len(self.grid) and 0 <= neighbor_cell[
                    1
                ] < len(
                    self.grid[0]
                ):  # Check if the neighbor cell is within the grid
                    if not visited[neighbor_cell[0]][
                        neighbor_cell[1]
                    ]:  # Check if the neighbor cell has not been visited
                        # Update distance to neighbor
                        distance = (
                            self.dijkstra_distance[current_cell[0]][current_cell[1]]
                            + cost_grid[neighbor_cell[0]][neighbor_cell[1]]
                        )
                        if abs(i + j) != 1:
                            distance += 0.414
                        # Update distance if it is smaller than the current distance
                        if (
                            distance
                            < self.dijkstra_distance[neighbor_cell[0]][neighbor_cell[1]]
                        ):
                            self.dijkstra_distance[neighbor_cell[0]][
                                neighbor_cell[1]
                            ] = distance

            # Mark current cell as visited
            visited[current_cell[0]][current_cell[1]] = True

            # Choose the next cell to visit
            min_distance = float("inf")
            for i in range(len(self.grid)):
                for j in range(len(self.grid[0])):
                    if (
                        not visited[i][j]
                        and self.dijkstra_distance[i][j] < min_distance
                    ):
                        min_distance = self.dijkstra_distance[i][j]
                        current_cell = [i, j]

            # Stop if all cells have been visited
            stop = True
            for i in range(len(self.grid)):
                if not all(visited[i]):
                    stop = False
                    break
            if stop:
                break

    def measure_start(self) -> None:
        """
        Saves the current x pos of the pedestrians in self.measure_x_positions and prints info about it at the console.
        """
        print("Starting measurement!")

        self.measure_x_positions = []
        for ped in self.pedestrians:
            self.measure_x_positions.append(ped.x)

    def measure_stop(self, chosen_file: str, stop_time_index: int) -> None:
        """
        Calculates the average speed of the pedestrians by using the step count as time and the x distance between the current position of the pedestrians
        and the x positions saved in self.measure_x_position which was set by an earlier method call of measure_start().

        :param chosen_file: name of the scenario file so that we know which file is analyzed in the console output
        :param stop_time_index: this methods prints different measurement texts on the console depending if its the first = 0 or second = 1 measurement.
        """
        average_speeds_measured = []
        average_speeds = []

        for i, ped in enumerate(self.pedestrians):
            average_speed = (ped.x - self.measure_x_positions[i]) / (
                self.measure_stop_step[stop_time_index] - self.measure_start_step
            )
            average_speeds_measured.append(average_speed * 3.5 * 0.4)
            average_speeds.append(ped.speed * 3.5 * 0.4)

        if stop_time_index == 0:
            print(f"\t{chosen_file}:")
            print(
                f"\t\tAverage speed at the first measuring point = {sum(average_speeds_measured) / len(average_speeds_measured)}"
            )
        elif stop_time_index == 1:
            print(
                f"\t\tAverage speed at the second measuring point = {sum(average_speeds_measured) / len(average_speeds_measured)}"
            )
            print(
                f"\t\tAverage of the speed values from all pedestrians = {sum(average_speeds) / len(average_speeds)}"
            )

    def has_valid_measure_parameters(self) -> bool:
        """
        Returns true if all measurement values for the scenario are valid or false otherwise. This is used to prevent measuring scenarios with invalid
        input values.
        """
        return (
            self.measure_start_step < self.measure_stop_step[0]
            and self.measure_start_step < self.measure_stop_step[1]
            and self.measure_start_step != -1
            and self.measure_stop_step[0] != -1
            and self.measure_stop_step[1] != -1
        )
