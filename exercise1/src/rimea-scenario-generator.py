import json


def rimea_bottleneck_scenario(dijkstra):
    '''Generates a scenario for the Rimea bottleneck scenario.
    '''

    scenario_data = {}
    num_pedestrians = 50

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
        
        scenario_data["pedestrians"].append({"x": x_val, "y": y_val, "dijkstra": dijkstra})
        if i % 5 == 0:
            x_val += 1
            y_val = 0
        else:
            y_val += 1

    # Add targets
    scenario_data["targets"].append({"x": 5, "y": 24})

    # Add obstacles
    for i in range(0,5):
        #scenario_data["obstacles"].append({"y": i, "x": 5})
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



rimea_bottleneck_scenario(dijkstra=False)
rimea_bottleneck_scenario(dijkstra=True)