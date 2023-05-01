"""
Main module for running the simulation and managing GUI.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""

import sys

import pygame
from constants import BACKGROUND_COLOR
from file_dialog import FileDialog
from grid import Grid
from helper_functions import increment_time_step_by_one, initialize_buttons
from scenario_loader import load_scenario
from visualisation import draw_elapsed_time, draw_step_counter

# Variable for Play/Pause button
DO_STEP_EVERY_X_MILLISECONDS = 300


def main() -> None:
    """
    Main function to initialize and run the crowd simulation.
    """
    # Load the initial scenario
    width: int
    height: int
    grid: Grid
    width, height, grid = load_scenario("scenarios/scenario-default.json")
    chosen_file = "scenarios/scenario-default.json"

    pygame.init()
    screen = pygame.display.set_mode((max(width, 700), max(height, 500)))
    pygame.display.set_caption("Crowd Simulation")

    # Initialize buttons
    next_step_button, choose_scenario_button, toggle_loop_button = initialize_buttons(
        height
    )
    loop_flag = False

    # Initialize the file dialog
    file_dialog: FileDialog = FileDialog("./scenarios", ".json", screen)

    step_count: int = 0

    # Define a custom event
    TIMER_EVENT = pygame.USEREVENT + 1

    elapsed_time = 0

    # Set the timer to generate the custom event every x milliseconds (1 second = 1000 milliseconds)
    pygame.time.set_timer(TIMER_EVENT, DO_STEP_EVERY_X_MILLISECONDS)

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
                        screen = pygame.display.set_mode(
                            (max(width, 700), max(height, 500))
                        )
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
                        step_count = increment_time_step_by_one(
                            grid, step_count, chosen_file
                        )
                    # Check if buttons are clicked
                    if toggle_loop_button.is_clicked(event.pos):
                        loop_flag = not loop_flag

                        # Initialize the clock and elapsed time variables
                        clock = pygame.time.Clock()

                        if loop_flag:
                            toggle_loop_button.set_text("Pause")
                        else:
                            toggle_loop_button.set_text("Simulate")

                    choose_scenario_button.on_click(event)

        # Draw elements on the screen
        screen.fill(BACKGROUND_COLOR)

        # Update the elapsed time
        if loop_flag:
            elapsed_time += clock.tick(60) / 1000.0

        if choose_scenario_button.choosing_scenario:
            file_dialog.draw()
        else:
            grid.draw(screen)
            next_step_button.draw(screen)
            choose_scenario_button.draw(screen)
            toggle_loop_button.draw(screen)
            draw_step_counter(
                screen, step_count, max(700, width) - 10, max(500, height) - 10
            )
            draw_elapsed_time(screen, elapsed_time, 0, height - 5)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    main()
