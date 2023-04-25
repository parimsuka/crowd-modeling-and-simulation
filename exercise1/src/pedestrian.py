import numpy as np


class Pedestrian:

    def __init__(self, x, y, speed=1):
        self.x = x
        self.y = y
        if 0 <= speed <= 1:
            self.speed = speed
        else:
            raise ValueError("A pedestrian speed >1 is not supported, since it might lead to skipping cells. "
                             "Instead, reduce the speed of all other pedestrians accordingly.")
        # self.waittime = 0  # TODO: check if we still want this

    def get_position(self):
        return self.x, self.y

    def get_grid_cell(self, grid):
        return grid[int(self.x)][int(self.y)]

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        return self.x, self.y

    def move_target_direction(self, target_x, target_y, distance, grid):
        dx, dy = self.find_best_move_cell(target_x, target_y, distance, grid)
        return self.move(dx, dy)

    def find_closest_target(self, targets):
        closest_target = None
        closest_target_distance = np.inf

        for target in targets:
            target_vector = [target[0] - self.x, target[1] - self.y]
            target_distance = np.linalg.norm(target_vector)
            if target_distance < closest_target_distance:
                closest_target_distance = target_distance
                closest_target = target

        return closest_target

    def move_to_closest_target(self, targets, grid):
        closest_target = self.find_closest_target(targets)
        if closest_target is None:
            return

        return self.move_target_direction(closest_target[0]+0.5, closest_target[1]+0.5, self.speed, grid)

    def get_move_deltas(self, target_x, target_y, distance):
        target_vector = [target_x - self.x, target_y - self.y]
        target_vector_norm = np.linalg.norm(target_vector)

        dx = target_vector[0] * distance / target_vector_norm
        dy = target_vector[1] * distance / target_vector_norm

        return dx, dy, target_vector_norm

    def find_best_move_cell(self, target_x, target_y, distance, grid):
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

        # Target cell is neither current cell nor empty
        # -> Consider if other neighboring cells are closer to target than current position
        int_x = int(self.x)
        int_y = int(self.y)
        neighbor_cells = []
        for i in [int_x-1, int_x, int_x+1]:
            if 0 <= i < len(grid):
                for j in [int_y-1, int_y, int_y+1]:
                    if 0 <= j < len(grid[0]):
                        neighbor_cells.append((i+0.5, j+0.5, grid[i][j]))

        best_neighbor_cell = 0, 0  # If we don't find a cell, do not move
        best_neighbor_cell_distance = target_vector_norm

        for nc_x, nc_y, nc_value in neighbor_cells:
            if nc_value == 'E' or nc_value == 'R' or nc_value == 'T':
                # Cell is empty -> Check for distance to target
                distance_vector = [target_x - nc_x, target_y - nc_y]
                distance_vector_norm = np.linalg.norm(distance_vector)
                if distance_vector_norm < best_neighbor_cell_distance:
                    best_neighbor_cell_distance = distance_vector_norm
                    best_neighbor_cell = nc_x, nc_y

        dx, dy, _ = self.get_move_deltas(best_neighbor_cell[0], best_neighbor_cell[1], distance)
        return dx, dy





