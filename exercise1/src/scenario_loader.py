"""
JSON scenario loading module.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


import json

import numpy as np
from grid import Grid
from pedestrian import Pedestrian
from pedestriancolors import *

CELL_CENTER_OFFSET: float = (
    0.5  # offset used to make pedestrians start in the middle of the cell
)


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

    measure_start = -1
    if "measure_start" in scenario:
        measure_start = scenario["measure_start"]

    measure_stop = (-1, -1)
    if "measure_stop" in scenario:
        measure_stop = scenario["measure_stop"]

    compute_dijkstra_distance = False
    grid: Grid = Grid(
        scenario["grid_height"],
        scenario["grid_width"],
        scenario["cell_size"],
        measure_start,
        measure_stop,
    )

    # Add pedestrians
    for ped in scenario["pedestrians"]:
        x = ped["x"]
        y = ped["y"]

        dijkstra_used = False
        if "dijkstra" in ped:
            dijkstra_used = ped["dijkstra"]
            if dijkstra_used:
                compute_dijkstra_distance = True

        speed = 1
        if "speed" in ped:
            speed = ped["speed"]

        grid_color = PedestrianColors.P_BLUE
        if "color" in ped:
            grid_color = PedestrianColors.get_color_by_name(ped["color"])

        grid.add_pedestrian(
            Pedestrian(
                x + CELL_CENTER_OFFSET,
                y + CELL_CENTER_OFFSET,
                dijkstra_used,
                speed,
                grid_color,
            )
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
    if "obstacles" in scenario:
        for obs in scenario["obstacles"]:
            grid.add_obstacle(obs["x"], obs["y"])

    if compute_dijkstra_distance:
        for tgt in scenario["targets"]:
            grid.dijkstra(tgt["x"], tgt["y"])

    # Compute dijkstra distance grid considering all targets
    if compute_dijkstra_distance:
        for tgt in scenario["targets"]:
            grid.dijkstra(tgt["x"], tgt["y"])

    width: int = scenario["grid_width"] * scenario["cell_size"]
    height: int = scenario["grid_height"] * scenario["cell_size"] + 60

    return width, height, grid
