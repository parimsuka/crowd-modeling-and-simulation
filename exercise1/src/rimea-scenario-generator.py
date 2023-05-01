"""
Efficient JSON scenario creation module.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


import json
import math
import random as rd

import numpy as np
from constants import SPEED_TABLE
from pedestriancolors import PedestrianColors


def ped_can_fit_in_list(new_ped, pedestrians):
    for ped in pedestrians:
        if ped["x"] == new_ped["x"] and ped["y"] == new_ped["y"]:
            print("Position taken.")
            return False
    return True


def normalize_pedestrian_speeds(pedestrian_speeds):
    """
    Normalize the pedestrian speeds based on the maximum speed.
    :param pedestrian_speeds: A list of pedestrian speeds.
    :return: A list of normalized and rounded pedestrian speeds.
    """

    # Get the maximum speed from the pedestrian speeds
    max_speed = max(pedestrian_speeds)

    # Normalize the pedestrian speeds based on the maximum speed
    normalized = [speed / max_speed for speed in pedestrian_speeds]

    # Round the normalized speeds to two decimals
    # rounded = [round(speed, 2) for speed in normalized]

    return normalized


def distribute_age_groups(num_pedestrians):
    """
    Uniformly distribute the pedestrians over the age groups.
    :param num_pedestrians: The total number of pedestrians to be added to the scenario.
    :return: A list of pedestrian speeds.
    """
    # Uniformly distribute the ages of pedestrians
    distribution = np.random.randint(low=0, high=15, size=num_pedestrians)

    # Convert the ages to speeds
    return [SPEED_TABLE[age] for age in distribution]


def rimea_test_scenario():
    """Generates a test scenario for the Rimea scenarios.
    To do:
    1) Initialize the dictionary (scenario_data)
    2) Add the grid parameters (grid_width, grid_height, cell_size)
    3) Add the pedestrians (x, y, dijkstra, speed) as a pedestrian list
    4) Add the targets (x, y, absorbable) as a target list
    5) Add the obstacles (x, y) as an obstacle list
    """

    scenario_data = {}

    scenario_data["grid_width"] = 20
    scenario_data["grid_height"] = 10
    scenario_data["cell_size"] = 30

    scenario_data["pedestrians"] = []

    for i in range(0, 10):
        scenario_data["pedestrians"].append({"x": 0, "y": i})

    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    scenario_data["targets"].append({"x": 5, "y": 5, "absorbable": True})

    # Dump as a json file
    with open("scenarios/test.json", "w") as f:
        json.dump(scenario_data, f, indent=4)


def rimea_bottleneck_scenario(dijkstra):
    """
    Generates a scenario for the Rimea bottleneck scenario.
    """

    scenario_data = {}
    num_pedestrians = 50
    pedestrian_speeds = distribute_age_groups(num_pedestrians)
    normalized_speeds = normalize_pedestrian_speeds(pedestrian_speeds)

    # Grid parameters
    scenario_data["grid_width"] = 25
    scenario_data["grid_height"] = 10
    scenario_data["cell_size"] = 30

    scenario_data["pedestrians"] = []
    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    # Add pedestrians
    x_val = 0
    y_val = 0

    for i in range(1, num_pedestrians + 1):
        scenario_data["pedestrians"].append(
            {
                "x": x_val,
                "y": y_val,
                "dijkstra": dijkstra,
                "speed": 1,
                "color": "P-Yellow",
            }
        )
        if i % 5 == 0:
            y_val += 1
            x_val = 0
        else:
            x_val += 1

    # Add targets
    scenario_data["targets"].append({"x": 24, "y": 0, "absorbable": True})

    # Add obstacles
    for i in range(0, 5):
        scenario_data["obstacles"].append({"x": 10, "y": i})
        scenario_data["obstacles"].append({"x": 15, "y": i})
        scenario_data["obstacles"].append({"x": i + 10, "y": 4})
        scenario_data["obstacles"].append({"x": i + 10, "y": 6})
        if i + 6 < 10:
            scenario_data["obstacles"].append({"x": 10, "y": i + 6})
            scenario_data["obstacles"].append({"x": 15, "y": i + 6})

    # Save scenario
    if dijkstra:
        with open(
            "scenarios/Task 5/rimea_4/rimea-task4-bottleneck-w-dijkstra.json", "w"
        ) as f:
            json.dump(scenario_data, f, indent=4)
    else:
        with open(
            "scenarios/Task 5/rimea_4/rimea-task4-bottleneck-wo-dijkstra.json", "w"
        ) as f:
            json.dump(scenario_data, f, indent=4)


def scenario_1():
    """
    Generates the json files that are used for RiMEA scenario 1 simulations.
    """
    scenario_data = {}

    scenario_data["grid_width"] = 100
    scenario_data["grid_height"] = 5
    scenario_data["cell_size"] = 11

    scenario_data["pedestrians"] = []
    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    # Add pedestrian
    scenario_data["pedestrians"].append({"x": 1, "y": 2, "dijkstra": False})

    # Add target
    scenario_data["targets"].append({"x": 99, "y": 2, "absorbable": True})

    # Add obstacles
    for i in range(100):
        scenario_data["obstacles"].append({"x": i, "y": 0})

    for i in range(100):
        scenario_data["obstacles"].append({"x": i, "y": 4})

    for i in range(5):
        scenario_data["obstacles"].append({"x": 0, "y": i})

    with open(f"scenarios/Task 5/rimea_1/rimea_1.json", "w") as f:
        json.dump(scenario_data, f, indent=4)


def scenario_4():
    """
    Generates the json files that are used for RiMEA scenario 4 simulations.

    Generates for every density value in DENSITIES a json file which resembles one scenario.
    Speeds are taken from a random uniform distribution.
    Colors of the pedestrians are given based on the rolled speed.
    Pedestrian spawning positions are randomly distributed in the spawning area.
    The measurement values measure_start and measure_stop are set accordingly.
    """
    DENSITIES: list[float] = [0.5, 1, 2, 3, 4, 5, 6]
    MAX_POSSIBLE_DENSITY: float = (
        6.25  # 0.4m * 0.4m = 0.16m^2 and then 1m^2 / 0.16m^2 = 6.25
    )
    MAX_CELL_COUNT: int = 1000  # = 40 * 25
    MIN_SPEED = 3.0  # 1.2m/s / 0.4m/cell (cell width) = 3.0 cells/s
    MAX_SPEED = 3.5  # 1.4m/s / 0.4m/cell (cell width) = 3.5 cells/s

    scenario_data = {}

    scenario_data["grid_width"] = 540
    scenario_data["grid_height"] = 25
    scenario_data["cell_size"] = 5
    scenario_data[
        "measure_start"
    ] = 35  # This represents waiting 10 seconds at the beginning
    scenario_data["measure_stop"] = (
        205,
        228,
    )  # This represents the measuring points in time. After 205 and 228 steps
    scenario_data["obstacles"] = []
    scenario_data["targets"] = []

    for i in range(25):
        scenario_data["targets"].append({"x": 539, "y": i, "absorbable": False})

    for density in DENSITIES:
        num_ped_for_density = MAX_CELL_COUNT * (density / MAX_POSSIBLE_DENSITY)
        print(num_ped_for_density)

        scenario_data["pedestrians"] = []

        # Random Distribution
        for i in range(int(num_ped_for_density)):
            speed = rd.uniform(MIN_SPEED, MAX_SPEED) / MAX_SPEED

            color = PedestrianColors.P_GRAY
            if speed < 0.88:
                color = PedestrianColors.P_BLUE
            elif speed < 0.91:
                color = PedestrianColors.P_GREEN
            elif speed < 0.94:
                color = PedestrianColors.P_YELLOW
            elif speed < 0.97:
                color = PedestrianColors.P_ORANGE
            elif speed < 1.0:
                color = PedestrianColors.P_RED
            else:
                raise ValueError(
                    f"Speed should be between {0} and {1}, including borders"
                )

            # pedestrians can spawn in a 40x25 area at the start of the corridor. If we hit a spot that is already taken we reroll the position.
            ped = {
                "x": rd.randint(0, 39),
                "y": rd.randint(0, 24),
                "speed": speed,
                "color": color.name,
            }
            while not ped_can_fit_in_list(ped, scenario_data["pedestrians"]):
                ped = {
                    "x": rd.randint(0, 39),
                    "y": rd.randint(0, 24),
                    "speed": speed,
                    "color": color.name,
                }
            scenario_data["pedestrians"].append(ped)

        # Dump as a json file
        with open(
            f"scenarios/Task 5/rimea_4/rimea_4_density={density}_minSpeed={1.2}_maxSpeed={1.4}.json",
            "w",
        ) as f:
            json.dump(scenario_data, f, indent=4)


def scenario_6():
    """
    Generates the json files that are used for RiMEA scenario 6 simulations.
    """
    # Set the number of pedestrians to generate
    NUM_PEDESTRIANS_LIST = [5, 10, 20, 30, 40]
    SPEED = 1
    X_MIN: int = 0
    X_MAX: int = 15
    Y_MIN: int = 21
    Y_MAX: int = 23

    scenario_data = {}

    scenario_data["grid_width"] = 25
    scenario_data["grid_height"] = 25
    scenario_data["cell_size"] = 17

    scenario_data["pedestrians"] = []
    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    for i in range(25):
        scenario_data["obstacles"].append({"x": i, "y": 24})

    for i in range(25):
        scenario_data["obstacles"].append({"x": 24, "y": i})

    for i in range(21):
        scenario_data["obstacles"].append({"x": i, "y": 20})

    for i in range(21):
        scenario_data["obstacles"].append({"x": 20, "y": i})

    for i in range(3):
        scenario_data["targets"].append({"x": 24 - i - 1, "y": 0, "absorbable": False})

    for num_peds in NUM_PEDESTRIANS_LIST:
        # Uniform distribution of pedestrians
        scenario_data["pedestrians"] = []
        for i in range(num_peds):
            color = PedestrianColors.get_random_p_color()
            # Generate random x and y coordinates using uniform distribution
            ped = {
                "x": rd.randint(X_MIN, X_MAX),
                "y": rd.randint(Y_MIN, Y_MAX),
                "speed": SPEED,
                "color": color.name,
            }
            while not ped_can_fit_in_list(ped, scenario_data["pedestrians"]):
                ped = {
                    "x": rd.randint(X_MIN, X_MAX),
                    "y": rd.randint(Y_MIN, Y_MAX),
                    "speed": SPEED,
                    "color": color.name,
                }

            scenario_data["pedestrians"].append(ped)

        with open(
            f"scenarios/Task 5/rimea_6/rimea_6_N={num_peds}.json",
            "w",
        ) as f:
            json.dump(scenario_data, f, indent=4)


def scenario_7():
    """
    Generates the json files that are used for RiMEA scenario 7 simulations.

    Speeds are taken from SPEED_TABLE in constants.py and are normalized to our simulation.
    Colors of the pedestrians are given based on the rolled speed.
    Pedestrian spawning positions are in the bottom horizontal line of the grid.
    """
    scenario_data = {}

    scenario_data["grid_width"] = 50
    scenario_data["grid_height"] = 50
    scenario_data["cell_size"] = 9

    scenario_data["pedestrians"] = []
    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    nr_pedestrians = 50
    values_to_distribute = len(SPEED_TABLE)
    MIN_SPEED = 1.75  # 0.7m/s / 0.4m/cell (cell width) = 1.75 cells/s
    MAX_SPEED = 4  # 1.6m/s / 0.4m/cell (cell width) = 4 cells/s

    for i in range(50):
        scenario_data["targets"].append({"x": i, "y": 0})

    step_size = math.ceil(nr_pedestrians / values_to_distribute)

    # Initialize the array with 0 values
    speed_arr = np.zeros(nr_pedestrians)
    keys = list(SPEED_TABLE.keys())

    # Fill the array with the values from SPEED_TABLE using the step size
    for i in range(0, nr_pedestrians, step_size):
        for j in range(step_size):
            if i + j < nr_pedestrians:
                speed_arr[i + j] = SPEED_TABLE[keys[i // step_size]]

    speed_arr = [(speed / 0.4) / MAX_SPEED for speed in speed_arr]

    for i, speed in enumerate(speed_arr):
        color = PedestrianColors.P_GRAY
        if speed < 0.75:
            color = PedestrianColors.P_BLUE
        elif speed < 0.88:
            color = PedestrianColors.P_GREEN
        elif speed < 0.91:
            color = PedestrianColors.P_YELLOW
        elif speed < 0.94:
            color = PedestrianColors.P_ORANGE
        elif speed <= 1.0:
            color = PedestrianColors.P_RED
        else:
            raise ValueError(f"Speed should be between {0} and {1}, including borders")

        ped = {"x": i, "y": 49, "speed": speed, "color": color.name}
        scenario_data["pedestrians"].append(ped)

    with open(f"scenarios/Task 5/rimea_7/rimea_7.json", "w") as f:
        json.dump(scenario_data, f, indent=4)


# rimea_bottleneck_scenario(dijkstra=False)
# rimea_bottleneck_scenario(dijkstra=True)
# rimea_test_scenario()
scenario_1()
scenario_4()
scenario_6()
scenario_7()
