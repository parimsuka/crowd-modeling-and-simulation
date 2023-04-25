import json

from grid import Grid
from pedestrian import Pedestrian


def load_scenario(scenario_path, grid_size, cell_size):
    """
    Load a scenario from a JSON file and initialize the grid.

    :param scenario_path: Path to the scenario JSON file.
    :param grid_size: The size of the grid.
    :param cell_size: The size of each cell in the grid.
    :return: An initialized Grid object.
    """
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)

    grid = Grid(grid_size, cell_size)

    # Add pedestrians
    for ped in scenario['pedestrians']:
        grid.add_pedestrian(Pedestrian(ped['x']+0.5, ped['y']+0.5))

    # Add targets
    if 'targets' in scenario:
        for tgt in scenario['targets']:
            grid.add_target(tgt['x'], tgt['y'])
    else:
        raise ValueError("No targets found in the scenario")

    # Add obstacles
    for obs in scenario['obstacles']:
        grid.add_obstacle(obs['x'], obs['y'])

    return grid

