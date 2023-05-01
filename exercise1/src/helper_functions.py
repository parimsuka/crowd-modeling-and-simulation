"""
Module that contains constants used in the project.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


from grid import Grid
from visualisation import Button, ChooseScenarioButton


def increment_time_step_by_one(grid: Grid, step_count: int, chosen_file: str):
    """
    Goes one time step further in the simulation. It upgrades the grid state, the step_counter and checks for possible measurement events.

    :param grid: Simulation grid where the current state is saved.
    :param step_count: Current step count of the simulation.
    :param chosen_file: Name of the json file that is currently run.
    :return: the input_count incremented by 1
    """
    grid.update()
    step_count += 1

    if grid.has_valid_measure_parameters:
        if step_count == grid.measure_start_step:
            grid.measure_start()
        if step_count == grid.measure_stop_step[0]:
            grid.measure_stop(chosen_file, 0)
        if step_count == grid.measure_stop_step[1]:
            grid.measure_stop(chosen_file, 1)

    return step_count


def initialize_buttons(height):
    # Initialize buttons
    next_step_button: Button = Button(
        10, height - 60, 100, 40, "Next Step", (200, 200, 200)
    )
    choose_scenario_button: ChooseScenarioButton = ChooseScenarioButton(
        120, height - 60, 200, 40, "Choose Scenario", (200, 200, 200)
    )
    toggle_loop_button = Button(330, height - 60, 100, 40, "Play", (200, 200, 200))

    return next_step_button, choose_scenario_button, toggle_loop_button
