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


def draw_step_counter(screen, step_count, x, y):
    """
    Draw the step counter text on the screen at the given position.
    :param screen: The surface to draw the step counter text on.
    :param step_count: The current step count.
    :param x: The x-coordinate of the bottom-right corner of the step counter text.
    :param y: The y-coordinate of the bottom-right corner of the step counter text.
    """
    font = pygame.font.Font(pygame.font.get_default_font(), 20)
    text = f"Steps: {step_count}"
    text_render = font.render(text, True, (0, 0, 0))
    text_rect = text_render.get_rect()
    text_rect.bottomright = (x, y)
    screen.blit(text_render, text_rect)
