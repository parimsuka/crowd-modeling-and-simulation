import sys

import pygame

from button import Button, ChooseScenarioButton
from constants import BACKGROUND_COLOR
from file_dialog import FileDialog
from scenario_loader import load_scenario
from grid import Grid
from utils import draw_step_counter, draw_elapsed_time

def increment_time_step_by_one(grid: Grid, step_count: int, chosen_file: str):
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


def main() -> None:
    """
    Main function to initialize and run the crowd simulation.
    """
    # Load the initial scenario
    width: int
    height: int
    grid: Grid
    width, height, grid = load_scenario("scenarios/scenario-default.json")

    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Crowd Simulation")

    # Initialize buttons
    next_step_button, choose_scenario_button, toggle_loop_button = initialize_buttons(
        height
    )
    loop_flag = False

    # Initialize the file dialog
    file_dialog: FileDialog = FileDialog("./scenarios", ".json", screen)

    step_count: int = 0

    # Initialize the clock and elapsed time variables
    clock = pygame.time.Clock()
    elapsed_time = 0
    # Define a custom event
    TIMER_EVENT = pygame.USEREVENT + 1

    # Set the timer to generate the custom event every 5000 milliseconds (5 seconds)
    pygame.time.set_timer(TIMER_EVENT, 200)

    # Main game loop
    while True:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == TIMER_EVENT and loop_flag:
                # Call your function here
                step_count = increment_time_step_by_one(grid, step_count, chosen_file)
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the file dialog is open and a file is clicked
                if choose_scenario_button.choosing_scenario:
                    chosen_file: str = file_dialog.get_clicked_file(event.pos)
                    if chosen_file is not None:
                        # Load the chosen scenario and reset step count
                        width, height, grid = load_scenario(chosen_file)
                        screen = pygame.display.set_mode((width, height))
                        pygame.display.set_caption("Crowd Simulation")

                        # Initialize buttons
                        (
                            next_step_button,
                            choose_scenario_button,
                            toggle_loop_button,
                        ) = initialize_buttons(height)

                        step_count = 0
                        elapsed_time = 0
                        choose_scenario_button.choosing_scenario = False
                else:
                    # Check if buttons are clicked
                    if next_step_button.is_clicked(event.pos):
                        step_count = increment_time_step_by_one(grid, step_count, chosen_file)
                    # Check if buttons are clicked
                    if toggle_loop_button.is_clicked(event.pos):
                        loop_flag = not loop_flag
                        if loop_flag:
                            toggle_loop_button.set_text("Pause")
                        else:
                            toggle_loop_button.set_text("Play")
                        
                    choose_scenario_button.on_click(event)

        # Draw elements on the screen
        screen.fill(BACKGROUND_COLOR)

        # Update the elapsed time
        elapsed_time += clock.tick(60) / 1000.0

        if choose_scenario_button.choosing_scenario:
            screen = pygame.display.set_mode((1000, 700)) # TODO Tali fix this? Screen is fluckering
            file_dialog.draw()
        else:
            grid.draw(screen)
            next_step_button.draw(screen)
            choose_scenario_button.draw(screen)
            toggle_loop_button.draw(screen)
            draw_step_counter(screen, step_count, width - 10, height - 10)
            draw_elapsed_time(screen, elapsed_time, 0, height - 5)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    main()
