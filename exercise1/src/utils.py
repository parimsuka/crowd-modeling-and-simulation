import pygame


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
