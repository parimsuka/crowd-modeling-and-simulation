"""
Module that contains constants used in the project.

author: Sena Korkut, PaRim Suka, Simon Blöchinger, Ricardo Kraft, Talwinder Singh
"""


# Constants for the crowd simulation
BACKGROUND_COLOR: tuple[int, int, int] = (
    220,
    220,
    220,
)  # The background color of the window
# EMPTY_CELL_COLOR: tuple[int, int, int] = (0, 0, 0)  # The color of an empty cell
EMPTY_CELL_COLOR: tuple[int, int, int] = (255, 255, 255)  # The color of an empty cell
PEDESTRIAN_COLOR: tuple[int, int, int] = (30, 144, 255)  # The color of a pedestrian
OBSTACLE_COLOR: tuple[int, int, int] = (0, 128, 0)  # The color of an obstacle
TARGET_COLOR: tuple[int, int, int] = (255, 0, 0)  # The color of a target
TRACE_COLOR: tuple[int, int, int] = (
    135,
    206,
    250,
)  # The color of a trace left by a pedestrian

# Table of pedestrian speed based on age
# Key values correspond to the age of the pedestrian i.e.
# 0: 10 years old, 1: 15 years old, 2: 20 years old, 3: 25 years old, 4: 30 years old, 5: 35 years old, 6: 40 years old etc.
SPEED_TABLE: dict[int, float] = {
    0: 1.2,
    1: 1.4,
    2: 1.6,
    3: 1.55,
    4: 1.52,
    5: 1.5,
    6: 1.47,
    7: 1.45,
    8: 1.4,
    9: 1.3,
    10: 1.25,
    11: 1.2,
    12: 1.1,
    13: 0.9,
    14: 0.7,
}
