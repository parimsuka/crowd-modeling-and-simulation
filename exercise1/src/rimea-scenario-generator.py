import json
import numpy as np
from constants import SPEED_TABLE


def normalize_pedestrian_speeds(pedestrian_speeds):
    '''
    Normalize the pedestrian speeds based on the maximum speed.
    :param pedestrian_speeds: A list of pedestrian speeds.
    :return: A list of normalized and rounded pedestrian speeds.'''

    #Get the maximum speed from the pedestrian speeds
    max_speed = max(pedestrian_speeds)

    #Normalize the pedestrian speeds based on the maximum speed
    normalized = [ speed/max_speed for speed in pedestrian_speeds]

    #Round the normalized speeds to two decimals
    #rounded = [round(speed, 2) for speed in normalized]

    return normalized


def distribute_age_groups(num_pedestrians):
    '''
    Uniformly distribute the pedestrians over the age groups.
    :param num_pedestrians: The total number of pedestrians to be added to the scenario.
    :return: A list of pedestrian speeds.
    '''
    # Uniformly distribute the ages of pedestrians
    distribution = np.random.randint(low=0, high=15, size=num_pedestrians)

    # Convert the ages to speeds
    return [SPEED_TABLE[age] for age in distribution]


def rimea_test_scenario():

    '''Generates a test scenario for the Rimea scenarios.
    To do: 
    1) Initialize the dictionary (scenario_data)
    2) Add the grid parameters (grid_width, grid_height, cell_size)
    3) Add the pedestrians (x, y, dijkstra, speed) as a pedestrian list
    4) Add the targets (x, y, absorbable) as a target list
    5) Add the obstacles (x, y) as an obstacle list)'''

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


    #Dump as a json file
    with open("scenarios/test.json", "w") as f:
            json.dump(scenario_data, f, indent=4)


def rimea_bottleneck_scenario(dijkstra):
    '''Generates a scenario for the Rimea bottleneck scenario.
    '''

    scenario_data = {}
    num_pedestrians = 50
    pedestrian_speeds = distribute_age_groups(num_pedestrians)
    normalized_speeds = normalize_pedestrian_speeds(pedestrian_speeds)

    # Grid parameters
    scenario_data["grid_width"] = 10
    scenario_data["grid_height"] = 25
    scenario_data["cell_size"] = 30

    scenario_data["pedestrians"] = []
    scenario_data["targets"] = []
    scenario_data["obstacles"] = []

    # Add pedestrians
    x_val = 0
    y_val = 0

    for i in range(1,num_pedestrians+1):
        
        scenario_data["pedestrians"].append({"x": x_val, "y": y_val, "dijkstra": dijkstra, "speed": normalized_speeds[i-1]})
        if i % 5 == 0:
            x_val += 1
            y_val = 0
        else:
            y_val += 1

    # Add targets
    scenario_data["targets"].append({"x": 5, "y": 24})

    # Add obstacles
    for i in range(0,5):
        scenario_data["obstacles"].append({"x": i, "y": 10})
        scenario_data["obstacles"].append({"x": i, "y": 15})
        scenario_data["obstacles"].append({"x": 4, "y": i+10})
        scenario_data["obstacles"].append({"x": 6, "y": i+10})
        if i + 6 < 10:
            scenario_data["obstacles"].append({"x": i+6, "y": 10})
            scenario_data["obstacles"].append({"x": i+6, "y": 15})

    # Save scenario
    if dijkstra:
        with open("scenarios/rimea-task4-bottleneck-w-dijkstra.json", "w") as f:
            json.dump(scenario_data, f, indent=4)
    else:
        with open("scenarios/rimea-task4-bottleneck-wo-dijkstra.json", "w") as f:
            json.dump(scenario_data, f, indent=4)



#rimea_bottleneck_scenario(dijkstra=False)
#rimea_bottleneck_scenario(dijkstra=True)
rimea_test_scenario()