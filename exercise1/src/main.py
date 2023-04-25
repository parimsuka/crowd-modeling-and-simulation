import sys

import pygame

from button import Button, ChooseScenarioButton
from constants import WIDTH, HEIGHT, BACKGROUND_COLOR, GRID_SIZE, CELL_SIZE
from file_dialog import FileDialog
from scenario_loader import load_scenario
from utils import draw_step_counter


def main():
    """
    Main function to initialize and run the crowd simulation.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crowd Simulation")

    # Load the initial scenario
    grid = load_scenario("Scenarios/scenario-default.json", GRID_SIZE, CELL_SIZE)

    # Initialize buttons
    next_step_button = Button(10, HEIGHT - 60, 100, 40, "Next Step", (200, 200, 200))
    choose_scenario_button = ChooseScenarioButton(120, HEIGHT - 60, 200, 40, "Choose Scenario", (200, 200, 200))

    # Initialize the file dialog
    file_dialog = FileDialog("./Scenarios", ".json", screen)

    step_count = 0

    # Main game loop
    while True:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the file dialog is open and a file is clicked
                if choose_scenario_button.choosing_scenario:
                    chosen_file = file_dialog.get_clicked_file(event.pos)
                    if chosen_file is not None:
                        # Load the chosen scenario and reset step count
                        grid = load_scenario(chosen_file, GRID_SIZE, CELL_SIZE)
                        step_count = 0
                        choose_scenario_button.choosing_scenario = False
                else:
                    # Check if buttons are clicked
                    if next_step_button.is_clicked(event.pos):
                        grid.update()
                        step_count += 1

                    choose_scenario_button.on_click(event)

        # Draw elements on the screen
        screen.fill(BACKGROUND_COLOR)

        if choose_scenario_button.choosing_scenario:
            file_dialog.draw()
        else:
            grid.draw(screen)
            next_step_button.draw(screen)
            choose_scenario_button.draw(screen)
            draw_step_counter(screen, step_count, WIDTH - 10, HEIGHT - 10)

        # Update the display
        pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    main()
