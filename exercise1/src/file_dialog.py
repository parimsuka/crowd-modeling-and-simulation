"""
File and folder dialog management module.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""

import os

import pygame


class FileDialog:
    """
    A simple file dialog class to display and select files.
    """

    def __init__(
        self,
        directory: str,
        file_extension: str,
        screen: pygame.Surface,
        background_color: tuple[int, int, int] = (200, 200, 200),
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
        self.icon_size: tuple[int, int] = (20, 20)
        self.folder_icon: pygame.Surface = pygame.transform.scale(
            pygame.image.load("../Folder.png").convert_alpha(), self.icon_size
        )
        self.folder_stack: list[str] = [self.directory]
        self.font: pygame.font.Font = pygame.font.Font(
            pygame.font.get_default_font(), 20
        )
        self.items: list[str] = self._get_files()

    def _get_files(self) -> list[str]:
        """
        Get a list of files and folders in the specified directory.
        :return: A list of file and folder names.
        """
        current_directory = self.folder_stack[-1]
        files = [
            f
            for f in os.listdir(current_directory)
            if f.endswith(self.file_extension)
            or os.path.isdir(os.path.join(current_directory, f))
        ]
        if len(self.folder_stack) > 1:
            files.insert(0, "..")
        return files

    def draw(self) -> None:
        """
        Draw the file dialog on the given surface.
        """
        self.screen.fill(self.background_color)

        for i, item in enumerate(self.items):
            current_directory = self.folder_stack[-1]
            item_path = os.path.join(current_directory, item)
            is_folder = os.path.isdir(item_path) or item == ".."

            if is_folder and self.folder_icon:
                icon_rect = self.folder_icon.get_rect()
                icon_rect.topleft = (10, 10 + i * 25)
                self.screen.blit(self.folder_icon, icon_rect)

            text_color = (0, 0, 255) if is_folder else (0, 0, 0)
            text_render: pygame.Surface = self.font.render(item, True, text_color)
            text_rect: pygame.rect = text_render.get_rect()
            text_rect.topleft: tuple[int, int] = (40 if is_folder else 10, 10 + i * 25)
            self.screen.blit(text_render, text_rect)

    def get_clicked_file(self, pos: tuple[int, int]) -> str:
        """
        Get the file or folder that was clicked, given the mouse position.
        :param pos: A tuple containing the x and y coordinates of the mouse position.
        :return: The file path of the clicked file or None if no file was clicked.
        """
        for i, item in enumerate(self.items):
            if pygame.Rect(10, 10 + i * 25, 500, 20).collidepoint(pos):
                current_directory = self.folder_stack[-1]
                clicked_path = os.path.join(current_directory, item)

                if os.path.isfile(clicked_path) and item.endswith(self.file_extension):
                    return clicked_path
                elif os.path.isdir(clicked_path):
                    self.folder_stack.append(clicked_path)
                    self.items = self._get_files()
                    return None
                elif item == "..":
                    self.folder_stack.pop()
                    self.items = self._get_files()
                    return None

        return None
