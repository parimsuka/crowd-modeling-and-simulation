import sys

import pygame

from button import Button, ChooseScenarioButton
from constants import BACKGROUND_COLOR
from file_dialog import FileDialog
from scenario_loader import load_scenario
from utils import draw_step_counter
from grid import Grid


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
    next_step_button: Button = Button(10, height - 60, 100, 40, "Next Step", (200, 200, 200))
    choose_scenario_button: ChooseScenarioButton = ChooseScenarioButton(
        120, height - 60, 200, 40, "Choose Scenario", (200, 200, 200)
    )

    # Initialize the file dialog
    file_dialog: FileDialog = FileDialog("./scenarios", ".json", screen)

    step_count: int = 0

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
                    chosen_file: str = file_dialog.get_clicked_file(event.pos)
                    if chosen_file is not None:
                        # Load the chosen scenario and reset step count
                        width, height, grid = load_scenario(chosen_file)
                        screen = pygame.display.set_mode((width, height))
                        pygame.display.set_caption("Crowd Simulation")

                        # Initialize buttons
                        next_step_button = Button(
                            10, height - 60, 100, 40, "Next Step", (200, 200, 200)
                        )
                        choose_scenario_button = ChooseScenarioButton(
                            120,
                            height - 60,
                            200,
                            40,
                            "Choose Scenario",
                            (200, 200, 200),
                        )
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
            draw_step_counter(screen, step_count, width - 10, height - 10)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    main()
