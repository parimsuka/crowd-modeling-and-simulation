import numpy as np


class Pedestrian:
    """
    A class representing pedestrians on a 2d space.
    The pedestrian has attributes for its position and speed.
    The class contains methods to choose a movement target for the pedestrian and methods to move the pedestrian.
    """

    def __init__(self, x: float, y: float, absorbable: bool, speed=1):
        """

        :param x: x-coordinate of the pedestrian.
        :param y: y-coordinate of the pedestrian.
        :param speed: The speed of the pedestrian.
            Note: in order to prevent skipping cells, a speed >1 is not supported.
        """
        self.x = x
        self.y = y
        self.absorbable = absorbable
        if 0 <= speed <= 1:
            self.speed = speed
        else:
            raise ValueError("A pedestrian speed >1 is not supported, since it might lead to skipping cells. "
                             "Instead, reduce the speed of all other pedestrians accordingly.\n"
                             "A speed <0 is not supported either.")

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

    def move_target_direction(self, target_x: float, target_y: float, distance: float, grid: list[list[str]]) -> (float, float):
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
            target_vector = [target[0] - self.x, target[1] - self.y]
            target_distance = np.linalg.norm(target_vector)
            if target_distance < closest_target_distance:
                closest_target_distance = target_distance
                closest_target = target

        return closest_target

    def move_to_closest_target(self, targets: list[tuple[int, int]], grid: list[list[str]]) -> (float, float):
        """
        Moves the pedestrian to the closest target.

        :param targets: List of target (x,y) tuples.
        :param grid: The grid containing the environment. Used to avoid pathing into certain cell types.
        :return: New position (x,y) of the pedestrian.
        """
        closest_target = self.find_closest_target(targets)
        if closest_target is None:
            return 0, 0

        return self.move_target_direction(closest_target[0]+0.5, closest_target[1]+0.5, self.speed, grid)

    def get_move_deltas(self, target_x: float, target_y: float, distance: float) -> (float, float, float):
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

    def find_best_move_cell(self, target_x: float, target_y: float, distance: float, grid: list[list[str]]) -> (float, float):
        """
        Calculates the best move for the pedestrian to reach a certain target, taking into account that the pedestrian
        only enters Empty, Target or Trace cells.

        :param target_x: The x-coordinate of the target.
        :param target_y: The y-coordinate of the target.
        :param distance: The distance which the pedestrian will be moved.
        :param grid: The grid containing the environment. Used to avoid pathing into certain cell types.
        :return: A tuple containing delta-x and delta-y, the distances that the pedestrian will move in the
            x- and y-direction
        """
        # calculate the possible new position of the pedestrian, moving directly towards the target
        dx, dy, target_vector_norm = self.get_move_deltas(target_x, target_y, distance)
        x_new = self.x + dx
        y_new = self.y + dy

        # If target position is in current cell: Move to target position
        if int(x_new) == int(self.x) and int(y_new) == int(self.y):
            return dx, dy

        # If target cell is empty: Move to target position
        if grid[int(x_new)][int(y_new)] == 'E':
            return dx, dy

        # If target cell is Target: Move inside target (currently only absorbing targets are implemented)
        if grid[int(x_new)][int(y_new)] == 'T':
            return dx, dy

        # If target cell is empty (but there is a trace): Move to new target position
        if grid[int(x_new)][int(y_new)] == 'R':
            return dx, dy

        # If we are here, Target cell is neither current cell nor empty nor a trace cell
        # -> Consider if other neighboring cells are closer to target than current position

        # enumerate all neighbor cells on the grid
        int_x = int(self.x)
        int_y = int(self.y)
        neighbor_cells = []
        for i in [int_x-1, int_x, int_x+1]:
            if 0 <= i < len(grid):
                for j in [int_y-1, int_y, int_y+1]:
                    if 0 <= j < len(grid[0]):
                        neighbor_cells.append((i+0.5, j+0.5, grid[i][j]))

        # Initial target position: our current position -> No better position: No movement
        best_neighbor_cell = self.x, self.y
        # Initial comparison: our current distance to our current target
        best_neighbor_cell_distance = target_vector_norm

        for nc_x, nc_y, nc_value in neighbor_cells:
            if nc_value == 'E' or nc_value == 'R' or nc_value == 'T':
                # Cell is empty (or Trace or Target) -> Check distance of this cell to target
                distance_vector = [target_x - nc_x, target_y - nc_y]
                distance_vector_norm = np.linalg.norm(distance_vector)

                # If the new cell has a smaller distance to the target: update the best neighbor cell (and distance)
                if distance_vector_norm < best_neighbor_cell_distance:
                    best_neighbor_cell_distance = distance_vector_norm
                    best_neighbor_cell = nc_x, nc_y

        # calculate the new move deltas to move the pedestrian towards the (new) target best_neighbor_cell
        dx, dy, _ = self.get_move_deltas(best_neighbor_cell[0], best_neighbor_cell[1], distance)
        return dx, dy
