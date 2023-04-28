import os
import pygame


class FileDialog:
    """
    A simple file dialog class to display and select files.
    """
    def __init__(
        self, directory: str, file_extension: str, screen: pygame.Surface, background_color: tuple[int, int, int] = (200, 200, 200)
    ) -> None:
        """
        Initialize a FileDialog instance.
        :param directory: The directory to display the files from.
        :param file_extension: The file extension of the files to display.
        :param screen: The surface to draw the file dialog on.
        :param background_color: The background color of the file dialog. Default is (200, 200, 200).
        """
        self.directory: str = directory
        self.file_extension: str = file_extension
        self.screen: pygame.Surface = screen
        self.background_color: tuple[int, int, int] = background_color
        self.files: list[str] = self._get_files()
        self.font: pygame.font.Font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def _get_files(self) -> list[str]:
        """
        Get a list of files in the specified directory with the specified file extension.
        :return: A list of file names.
        """
        return [
            f for f in os.listdir(self.directory) if f.endswith(self.file_extension)
        ]

    def draw(self) -> None:
        """
        Draw the file dialog on the given surface.
        """
        self.screen.fill(self.background_color)

        for i, file in enumerate(self.files):
            text_render: pygame.Surface = self.font.render(file, True, (0, 0, 0))
            text_rect: pygame.rect = text_render.get_rect()
            text_rect.topleft: tuple[int, int] = (10, 10 + i * 25)
            self.screen.blit(text_render, text_rect)

    def get_clicked_file(self, pos: tuple[int, int]) -> str:
        """
        Get the file that was clicked, given the mouse position.
        :param pos: A tuple containing the x and y coordinates of the mouse position.
        :return: The file path of the clicked file or None if no file was clicked.
        """
        for i, file in enumerate(self.files):
            if pygame.Rect(10, 10 + i * 25, 500, 20).collidepoint(pos):
                return os.path.join(self.directory, file)
        return None
