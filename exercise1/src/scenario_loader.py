import json

from grid import Grid
from pedestrian import Pedestrian


def load_scenario(scenario_path):
    """
    Load a scenario from a JSON file and initialize the grid.

    :param scenario_path: Path to the scenario JSON file.
    :return:
    -The width of the complete application window.
    -The height of the complete application window.
    -An initialized Grid object.
    """
    with open(scenario_path, "r") as f:
        scenario = json.load(f)

    grid = Grid(scenario["grid_height"], scenario["grid_width"], scenario["cell_size"])
    absorbable = scenario["absorbable"]
    # Add pedestrians
    for ped in scenario["pedestrians"]:
        grid.add_pedestrian(Pedestrian(ped["x"] + 0.5, ped["y"] + 0.5, absorbable))

    # Add targets
    if "targets" in scenario:
        for tgt in scenario["targets"]:
            grid.add_target(tgt["x"], tgt["y"])
    else:
        raise ValueError("No targets found in the scenario")

    # Add obstacles
    for obs in scenario["obstacles"]:
        grid.add_obstacle(obs["x"], obs["y"])

    width = scenario["grid_width"] * scenario["cell_size"]
    height = scenario["grid_height"] * scenario["cell_size"] + 60

    return width, height, grid
