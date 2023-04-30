import json
import numpy as np

from pedestriancolors import *

from grid import Grid
from pedestrian import Pedestrian


def load_scenario(scenario_path: str) -> tuple[int, int, Grid]:
    """
    Load a scenario from a JSON file and initialize the grid.

    :param scenario_path: Path to the scenario JSON file.
    :return:
    - The width of the complete application window.
    - The height of the complete application window.
    - An initialized Grid object.
    """
    with open(scenario_path, "r") as f:
        scenario: dict = json.load(f)

    if "cell_size" not in scenario:
        # Choose an appropriate cell size
        height_limit: int = 900  # height limit of 900px
        width_limit: int = 1600  # width limit of 1600px
        scenario["cell_size"] = int(
            np.minimum(
                height_limit / scenario["grid_height"],
                width_limit / scenario["grid_width"],
            )
        )

    grid: Grid = Grid(
        scenario["grid_height"], scenario["grid_width"], scenario["cell_size"]
    )

    # Add pedestrians
    for ped in scenario["pedestrians"]:
        x = ped["x"]
        y = ped["y"]

        dijkstra_used = False
        if "dijkstra" in ped:
            dijkstra_used = ped["dijkstra"]

        speed = 1
        if "speed" in ped:
            speed = ped["speed"]

        grid_color = PedestrianColors.P_BLUE
        if "color" in ped:
            grid_color = PedestrianColors.get_color_by_name(ped["color"])
            print(grid_color)

        grid.add_pedestrian(
            Pedestrian(x + 0.5, y + 0.5, dijkstra_used, speed, grid_color)
        )

    # Add targets
    if "targets" in scenario:
        for tgt in scenario["targets"]:
            if "absorbable" in tgt:
                grid.add_target(tgt["x"], tgt["y"], tgt["absorbable"])
            else:
                grid.add_target(tgt["x"], tgt["y"])
    else:
        raise ValueError("No targets found in the scenario")

    # Add obstacles
    for obs in scenario["obstacles"]:
        grid.add_obstacle(obs["x"], obs["y"])

    width: int = scenario["grid_width"] * scenario["cell_size"]
    height: int = scenario["grid_height"] * scenario["cell_size"] + 60

    return width, height, grid
