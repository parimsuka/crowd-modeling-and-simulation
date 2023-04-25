import pygame


class Button:
    """
    A simple button class with text to be used in a Pygame application.
    """
    def __init__(self, x, y, width, height, text, color):
        """
        Initialize a Button instance.
        :param x: The x-coordinate of the top-left corner of the button.
        :param y: The y-coordinate of the top-left corner of the button.
        :param width: The width of the button.
        :param height: The height of the button.
        :param text: The text displayed on the button.
        :param color: The color of the button.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color

    def draw(self, screen):
        """
        Draw the button on the given surface.
        :param screen: The surface to draw the button on.
        """
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_render = font.render(self.text, True, (0, 0, 0))
        text_rect = text_render.get_rect()
        text_rect.center = (self.x + self.width // 2, self.y + self.height // 2)
        screen.blit(text_render, text_rect)

    def is_clicked(self, pos):
        """
        Check if the button is clicked, given the mouse position.
        :param pos: A tuple containing the x and y coordinates of the mouse position.
        :return: True if the button is clicked, False otherwise.
        """
        x, y = pos
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


class ChooseScenarioButton(Button):
    """
    A subclass of the Button class that includes a callback function to be executed when the button is clicked.
    """
    def __init__(self, x, y, width, height, text, color):
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
        self.choosing_scenario = False

    def choose_scenario_callback(self):
        self.choosing_scenario = True

    def on_click(self, event):
        """
        Execute the callback function if the button is clicked.
        :param event: A Pygame event representing a mouse click.
        """
        if self.is_clicked(event.pos):
            self.choose_scenario_callback()