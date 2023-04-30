import numpy as np

from constants import PEDESTRIAN_COLOR


class Pedestrian:
    """
    A class representing pedestrians on a 2d space.
    The pedestrian has attributes for its position and speed.
    The class contains methods to choose a movement target for the pedestrian and methods to move the pedestrian.
    """

    def __init__(
        self, x: float, y: float, dijkstra_used=False, speed=1, color=PEDESTRIAN_COLOR
    ):
        """

        :param x: x-coordinate of the pedestrian.
        :param y: y-coordinate of the pedestrian.
        :param speed: The speed of the pedestrian.
            Note: in order to prevent skipping cells, a speed >1 is not supported.
        """
        self.x = x
        self.y = y
        self.dijkstra_used = dijkstra_used
        self.color = color
        if 0 <= speed <= 1:
            self.speed = speed
        else:
            raise ValueError(
                "A pedestrian speed >1 is not supported, since it might lead to skipping cells. "
                "Instead, reduce the speed of all other pedestrians accordingly.\n"
                "A speed <0 is not supported either."
            )

    def get_position(self) -> (float, float):
        """Returns the current position of the pedestrian as a float tuple."""
        return self.x, self.y

    def set_position(self, x, y):
        """
        Set the position of the pedestrian.

        :param x: The new x-coordinate of the pedestrian
        :param y: The new y-coordinate of the pedestrian
        """
        self.x = x
        self.y = y

    def move(self, dx: float, dy: float) -> (float, float):
        """
        Moves the pedestrian by delta-x in the x-direction and delta-y in the y-direction.

        :param dx: Delta-x.
        :param dy: Delta-y.
        :return: New position (x,y) of the pedestrian.
        """
        self.x += dx
        self.y += dy
        return self.x, self.y

    def move_target_direction(
        self, target_x: float, target_y: float, distance: float, grid: list[list[str]]
    ) -> (float, float):
        """
        Moves the pedestrian in the direction of a target for a distance. Only enters Empty, Trace and Target Cells.

        :param target_x: x-coordinate of the target.
        :param target_y: y-coordinate of the target.
        :param distance: The distance that the Pedestrian object will be moved.
        :param grid: The grid containing the environment. Used to avoid pathing into certain cell types.
        :return: New position (x,y) of the pedestrian.
        """
        dx, dy = self.find_best_move_cell(target_x, target_y, distance, grid)
        return self.move(dx, dy)

    def find_closest_target(self, targets: list[tuple[int, int]]) -> (int, int):
        """
        Finds the closest target to the pedestrian.

        :param targets: List of target (x,y) tuples.
        :return: (x,y) of the closest target.
        """
        closest_target = None
        closest_target_distance = np.inf

        # iterate over all possible targets to find the closest target
        for target in targets:
            target_vector = [target[0] + 0.5 - self.x, target[1] + 0.5 - self.y]
            target_distance = np.linalg.norm(target_vector)
            if target_distance < closest_target_distance:
                closest_target_distance = target_distance
                closest_target = target

        return closest_target

    def move_to_closest_target(
        self, targets: list[tuple[int, int]], grid: list[list[str]]
    ) -> (float, float):
        """
        Moves the pedestrian to the closest target.

        :param targets: List of target (x,y) tuples.
        :param grid: The grid containing the environment. Used to avoid pathing into certain cell types.
        :return: New position (x,y) of the pedestrian.
        """

        closest_target = self.find_closest_target(targets)
        if closest_target is None:
            return 0, 0

        if self.dijkstra_used:
            # Call dijsktra to find the shortest path
            path = self.dijkstra(closest_target[0], closest_target[1], grid)
            # print(path)

            # Make the resulting path the only available cells for the pedestrian
            new_grid = [["O" for _ in range(len(grid[0]))] for _ in range(len(grid))]
            for cell in path:
                if not (
                    grid[cell[0]][cell[1]] == "Ta"
                    or grid[cell[0]][cell[1]] == "Tn"
                    or grid[cell[0]][cell[1]].startswith("P")
                ):
                    new_grid[cell[0]][cell[1]] = "E"

            return self.move_target_direction(
                closest_target[0] + 0.5, closest_target[1] + 0.5, self.speed, new_grid
            )

        return self.move_target_direction(
            closest_target[0] + 0.5, closest_target[1] + 0.5, self.speed, grid
        )

    def get_move_deltas(
        self, target_x: float, target_y: float, distance: float
    ) -> (float, float, float):
        """
        Calculates delta-x and delta-y to move the pedestrian a certain distance towards the target.
        Also calculates the distance between the pedestrian and the target.

        :param target_x: The x-coordinate of the target.
        :param target_y: The y-coordinate of the target.
        :param distance: The distance which the pedestrian will be moved.
        :return: A tuple containing delta-x, delta-y and the distance between the pedestrian and the target.
        """
        # calculate the target_vector from the pedestrian to the target
        target_vector = [target_x - self.x, target_y - self.y]
        # calculate the length of the vector from the pedestrian to the target
        target_vector_norm = np.linalg.norm(target_vector)

        # calculate dx and dy to move the pedestrian `distance` towards the target
        if target_vector_norm != 0:
            dx = target_vector[0] * distance / target_vector_norm
            dy = target_vector[1] * distance / target_vector_norm
        else:
            dx, dy = 0, 0

        return dx, dy, target_vector_norm

    def find_best_move_cell(
        self,
        target_x: float,
        target_y: float,
        walking_distance: float,
        grid: list[list[str]],
    ) -> (float, float):
        """
        Calculates the best move for the pedestrian to reach a certain target, taking into account that the pedestrian
        only enters Empty, Target or Trace cells.

        :param target_x: The x-coordinate of the target.
        :param target_y: The y-coordinate of the target.
        :param walking_distance: The distance which the pedestrian will be moved.
        :param grid: The grid containing the environment. Used to avoid pathing into certain cell types.
        :return: A tuple containing delta-x and delta-y, the distances that the pedestrian will move in the
            x- and y-direction
        """
        # calculate the possible new position of the pedestrian, moving directly towards the target
        dx, dy, target_vector_norm = self.get_move_deltas(
            target_x, target_y, walking_distance
        )
        x_new = self.x + dx
        y_new = self.y + dy

        # If target position is in current cell: Move to target position
        if int(x_new) == int(self.x) and int(y_new) == int(self.y):
            return dx, dy

        # If target cell is empty: Move to target position
        if grid[int(x_new)][int(y_new)] == "E":
            return dx, dy

        # If target cell is absorbable Target: Move inside target
        if grid[int(x_new)][int(y_new)] == "Ta":
            return dx, dy

        # If target cell is empty (but there is a trace): Move to new target position
        if grid[int(x_new)][int(y_new)].startswith("R"):
            return dx, dy

        # If we are here, Target cell is neither current cell nor empty nor a trace cell
        # -> Consider if other neighboring cells are closer to target than current position
        reachable_cells = self.get_reachable_cells(grid)

        # Initial target position: our current position -> No better position: No movement
        best_neighbor_cell = self.x, self.y, 0
        # Initial comparison: our current distance to our current target
        best_neighbor_cell_distance = target_vector_norm

        for (
            rc_x,
            rc_y,
            rc_value,
            rc_contact_x,
            rc_contact_y,
            rc_distance,
        ) in reachable_cells:
            if (
                rc_value == "E"
                or rc_value.startswith("R")
                or rc_value == "Ta"
                or rc_value == "current"
            ):
                # Cell is empty or Trace or absorbableTarget or current
                # -> Calculate which of these cells is closest to our target (with respect to the distance to us)
                distance_cell_target = [
                    target_x - (rc_x + 0.5),
                    target_y - (rc_y + 0.5),
                ]
                distance_cell_target_norm = np.linalg.norm(distance_cell_target)

                # If the new cell has a smaller distance to the target: update the best neighbor cell (and distance)
                if distance_cell_target_norm < best_neighbor_cell_distance:
                    best_neighbor_cell_distance = distance_cell_target_norm
                    best_neighbor_cell = [rc_contact_x, rc_contact_y, walking_distance]
                    # Make sure we don't overshoot if we want to stay inside our cell.
                    if rc_value == "current":
                        best_neighbor_cell[2] = rc_distance

        # calculate the new move deltas to move the pedestrian towards the (new) target best_neighbor_cell
        dx, dy, _ = self.get_move_deltas(
            best_neighbor_cell[0], best_neighbor_cell[1], best_neighbor_cell[2]
        )
        return dx, dy

    def get_reachable_cells(
        self, grid: list[list[str]]
    ) -> list[list[int, int, str, float, float, float]]:
        """
        Searches all neighboring cells to find the reachable cells for a pedestrian with a respective speed.

        :param grid: The current grid.
        :return: List: [x, y, val, contact_point_x, contact_point_y, distance]
            x, y: The x- and y-coordinate of the reacable cell (top left corner)
            val: The grid status of the cell
            contact_point_x, contact_point_y: The x- and y-coordinate of the closest cell point to the pedestrian
            distance: The distance of the pedestrian to the closest cell point
        """
        int_x = int(self.x)
        int_y = int(self.y)

        # Variable containing all neighboring cells, starting with the current cell
        neighbor_cells = [[int_x, int_y, "current"]]

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
            cell_x = int_x + i
            cell_y = int_y + j
            if 0 <= cell_x < len(grid) and 0 <= cell_y < len(grid[0]):
                neighbor_cells.append([cell_x, cell_y, grid[cell_x][cell_y]])

        c = 1  # edge size of one cell, probably will always stay 1
        reachable_cells = []
        for x, y, val in neighbor_cells:
            # calculate where the pedestrian is with respect to the corner
            dx_s = x - self.x  # Distance to the smaller (left) side in x-direction
            dx_l = self.x - (
                x + c
            )  # Distance to the larger (right) side in x-direction
            dy_s = y - self.y  # Distance to the smaller (up)  side in y-direction
            dy_l = self.y - (y + c)  # Distance to the larger (down) side in y-direction

            # calculate the closest point of the cell to the pedestrian
            if dx_s > 0:
                contact_point_x = x
            elif dx_l > 0:
                contact_point_x = x + c
            else:
                contact_point_x = self.x

            if dy_s > 0:
                contact_point_y = y
            elif dy_l > 0:
                contact_point_y = y + c
            else:
                contact_point_y = self.y

            # calculate distance from pedestrian to closest cell edge/corner
            dx = np.max([dx_s, dx_l, 0])
            dy = np.max([dy_s, dy_l, 0])

            # Contact point for our current cell: middle of our current cell
            # Ensures that the pedestrian can reach all surrounding cells after this move
            if val == "current":
                contact_point_x = x + 0.5
                contact_point_y = y + 0.5
                dx = contact_point_x - self.x
                dy = contact_point_y - self.y

            distance = np.sqrt(dx * dx + dy * dy)

            if distance <= self.speed:
                reachable_cells.append(
                    [x, y, val, contact_point_x, contact_point_y, distance]
                )

        return reachable_cells

    def backtrack(self, target_x, target_y, distances):
        """
        Backtrack from the target cell to the start cell using the distance matrix
        :param target_x: x-coordinate of the target cell
        :param target_y: y-coordinate of the target cell
        :param distances: distance matrix
        :return: path from the target cell to the start cell
        """

        # Start from the last cell and backtrack to the first cell with the lowest cost cells
        path = [[target_x, target_y]]

        while True:
            # check up down left right - choose the direction that has the least distance
            neighbor_distances = []
            neighbor_cells = []

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
                neighbor_cell = [path[-1][0] + i, path[-1][1] + j]
                if 0 <= neighbor_cell[0] < len(distances) and 0 <= neighbor_cell[
                    1
                ] < len(distances[0]):
                    neighbor_cells.append(neighbor_cell)
                    neighbor_distances.append(
                        distances[neighbor_cell[0]][neighbor_cell[1]]
                    )

            path.append(neighbor_cells[np.argmin(neighbor_distances)])

            if path[-1][0] == int(self.x) and path[-1][1] == int(self.y):
                break

        return list(reversed(path))

    def dijkstra(self, target_x, target_y, grid):
        """
        Dijkstra algorithm to find the shortest path from the target cell to the start cell
        :param target_x: x-coordinate of the target cell
        :param target_y: y-coordinate of the target cell
        :param grid: the grid
        :return: the shortest path from the target cell to the start cell
        """

        # Initialize costs for each cell in the grid
        cost_grid = [[1 for _ in range(len(grid[0]))] for _ in range(len(grid))]

        # Obstacles have infinite cost (we don't want to go through them)
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == "O":
                    cost_grid[x][y] = float("inf")

        # Target and pedestrian's initial cell have 0 costs
        cost_grid[target_x][target_y] = 0
        cost_grid[int(self.x)][int(self.y)] = 0

        # Initialize the visited cells
        visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]

        # Initialize the distance from each cell to pedestrian, the distance for its own position is 0
        distances = [
            [float("inf") for _ in range(len(grid[0]))] for _ in range(len(grid))
        ]
        distances[int(self.x)][int(self.y)] = 0

        # Begin the search
        current_cell = [int(self.x), int(self.y)]
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
                if 0 <= neighbor_cell[0] < len(grid) and 0 <= neighbor_cell[1] < len(
                    grid[0]
                ):  # Check if the neighbor cell is within the grid
                    if not visited[neighbor_cell[0]][
                        neighbor_cell[1]
                    ]:  # Check if the neighbor cell has not been visited
                        # Update distance to neighbor
                        distance = (
                            distances[current_cell[0]][current_cell[1]]
                            + cost_grid[neighbor_cell[0]][neighbor_cell[1]]
                        )
                        if abs(i + j) != 1:
                            distance += 0.41
                        # Update distance if it is smaller than the current distance
                        if distance < distances[neighbor_cell[0]][neighbor_cell[1]]:
                            distances[neighbor_cell[0]][neighbor_cell[1]] = distance

            # Mark current cell as visited
            visited[current_cell[0]][current_cell[1]] = True

            # Choose the next cell to visit
            min_distance = float("inf")
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if not visited[i][j] and distances[i][j] < min_distance:
                        min_distance = distances[i][j]
                        current_cell = [i, j]

            # Stop if we have reached the target cell
            if current_cell[0] == target_x and current_cell[1] == target_y:
                break

        # Backtrack for the path
        return self.backtrack(target_x, target_y, distances)
