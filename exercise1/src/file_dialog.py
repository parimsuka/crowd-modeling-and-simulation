import os

import pygame


class FileDialog:
    """
    A simple file dialog class to display and select files.
    """

    def __init__(
        self, directory, file_extension, screen, background_color=(200, 200, 200)
    ):
        """
        Initialize a FileDialog instance.
        :param directory: The directory to display the files from.
        :param file_extension: The file extension of the files to display.
        :param screen: The surface to draw the file dialog on.
        :param background_color: The background color of the file dialog. Default is (200, 200, 200).
        """
        self.directory = directory
        self.file_extension = file_extension
        self.screen = screen
        self.background_color = background_color
        self.files = self._get_files()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def _get_files(self):
        """
        Get a list of files in the specified directory with the specified file extension.
        :return: A list of file names.
        """
        return [
            f for f in os.listdir(self.directory) if f.endswith(self.file_extension)
        ]

    def draw(self):
        """
        Draw the file dialog on the given surface.
        """
        self.screen.fill(self.background_color)

        for i, file in enumerate(self.files):
            text_render = self.font.render(file, True, (0, 0, 0))
            text_rect = text_render.get_rect()
            text_rect.topleft = (10, 10 + i * 25)
            self.screen.blit(text_render, text_rect)

    def get_clicked_file(self, pos):
        """
        Get the file that was clicked, given the mouse position.
        :param pos: A tuple containing the x and y coordinates of the mouse position.
        :return: The file path of the clicked file or None if no file was clicked.
        """
        for i, file in enumerate(self.files):
            if pygame.Rect(10, 10 + i * 25, 500, 20).collidepoint(pos):
                return os.path.join(self.directory, file)
        return None
