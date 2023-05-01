"""
Simulation visualization and GUI module.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


import pygame


class Button:
    """
    A simple button class with text to be used in a Pygame application.
    """

    def __init__(
        self, x: int, y: int, width: int, height: int, text: str, color: tuple
    ):
        """
        Initialize a Button instance.
        :param x: The x-coordinate of the top-left corner of the button.
        :param y: The y-coordinate of the top-left corner of the button.
        :param width: The width of the button.
        :param height: The height of the button.
        :param text: The text displayed on the button.
        :param color: The color of the button.
        """
        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.text: str = text
        self.color: tuple = color

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the button on the given surface.
        :param screen: The surface to draw the button on.
        """
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        font: pygame.font.FontType = pygame.font.Font(
            pygame.font.get_default_font(), 20
        )
        text_render: pygame.Surface = font.render(self.text, True, (0, 0, 0))
        text_rect: pygame.Rect = text_render.get_rect()
        text_rect.center: tuple[int, int] = (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )
        screen.blit(text_render, text_rect)

    def is_clicked(self, pos: tuple) -> bool:
        """
        Check if the button is clicked, given the mouse position.
        :param pos: A tuple containing the x and y coordinates of the mouse position.
        :return: True if the button is clicked, False otherwise.
        """
        x: float
        y: float
        x, y = pos
        return (
            self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
        )

    def set_text(self, new_text: str) -> None:
        self.text = new_text


class ChooseScenarioButton(Button):
    """
    A subclass of the Button class that includes a callback function to be executed when the button is clicked.
    """

    def __init__(
        self, x: int, y: int, width: int, height: int, text: str, color: tuple
    ):
        """
        Initialize a ChooseScenarioButton instance.
        :param x: The x-coordinate of the top-left corner of the button.
        :param y: The y-coordinate of the top-left corner of the button.
        :param width: The width of the button.
        :param height: The height of the button.
        :param text: The text displayed on the button.
        :param color: The color of the button.
        """
        super().__init__(x, y, width, height, text, color)
        self.choosing_scenario: bool = False

    def choose_scenario_callback(self) -> None:
        self.choosing_scenario = True

    def on_click(self, event: pygame.event.Event) -> None:
        """
        Execute the callback function if the button is clicked.
        :param event: A Pygame event representing a mouse click.
        """
        if self.is_clicked(event.pos):
            self.choose_scenario_callback()


def draw_rounded_rect(surface, color, rect, corner_radius):
    """
    Draw a rounded rectangle on the given surface with the specified color, rectangular dimensions, and corner radius.
    :param surface: The surface to draw the rounded rectangle on.
    :param color: The color of the rounded rectangle.
    :param rect: A pygame.Rect object defining the dimensions of the rectangle.
    :param corner_radius: The radius of the rounded corners.
    """
    pygame.draw.rect(surface, color, rect, border_radius=corner_radius)


def draw_step_counter(screen: pygame.Surface, step_count: int, x: int, y: int) -> None:
    """
    Draw the step counter text on the screen at the given position.
    :param screen: The surface to draw the step counter text on.
    :param step_count: The current step count.
    :param x: The x-coordinate of the bottom-right corner of the step counter text.
    :param y: The y-coordinate of the bottom-right corner of the step counter text.
    """
    font: pygame.font.Font = pygame.font.Font(pygame.font.get_default_font(), 20)
    text: str = f"Steps: {step_count}"
    text_render: pygame.Surface = font.render(text, True, (0, 0, 0))
    text_rect: pygame.Rect = text_render.get_rect()
    text_rect.bottomright: tuple[int, int] = (x, y)
    screen.blit(text_render, text_rect)


def draw_elapsed_time(screen, elapsed_time, x, y):
    """
    Draw the step counter text on the screen at the given position.
    :param screen: The surface to draw the step counter text on.
    :param step_count: The current step count.
    :param x: The x-coordinate of the bottom-right corner of the step counter text.
    :param y: The y-coordinate of the bottom-right corner of the step counter text.
    """
    # Render the elapsed time as text
    font = pygame.font.Font(None, 20)
    text = "Elapsed Time: {:.2f} seconds".format(elapsed_time)
    text_render = font.render(text, True, (0, 0, 0))
    text_rect = text_render.get_rect()
    text_rect.bottomleft = (x, y)
    screen.blit(text_render, text_rect)
    screen.blit(text_render, text_rect)
