from enum import Enum

class PedestrianColors(Enum):
    P_BLUE = ("P-Blue", (30, 144, 255), "R-Blue")
    R_BLUE = ("R-Blue", (135, 206, 250), None)
    
    P_RED = ("P-Red", (255, 0, 0), "R-Red")
    R_RED = ("R-Red", (255, 192, 203), None)
    
    P_ORANGE = ("P-Orange", (255, 165, 0), "R-Orange")
    R_ORANGE = ("R-Orange", (255, 218, 185), None)

    P_PURPLE = ("P-Purple", (128, 0, 128), "R-Purple")
    R_PURPLE = ("R-Purple", (218, 112, 214), None)
    
    P_GREEN = ("P-Green", (0, 128, 0), "R-Green")
    R_GREEN = ("R-Green", (144, 238, 144), None)

    P_BROWN = ("P-Brown", (139, 69, 19), "R-Brown")
    R_BROWN = ("R-Brown", (244, 164, 96), None)

    P_YELLOW = ("P-Yellow", (255, 255, 0), "R-Yellow")
    R_YELLOW = ("R-Yellow", (255, 255, 153), None)

    P_GRAY = ("P-Gray", (128, 128, 128), "R-Gray")
    R_GRAY = ("R-Gray", (192, 192, 192), None)
    
    def __init__(self, name, main_color, trace_name):
        self._name = name
        self._main_color = main_color
        self._trace_name = trace_name
        
    @property
    def name(self):
        return self._name
        
    @property
    def main_color(self):
        return self._main_color
    
    @property
    def trace_name(self):
        return self._trace_name
    
    @classmethod
    def get_color_by_name(cls, name: str) -> 'PedestrianColors':
        for color in cls:
            if color.name == name:
                return color
        raise ValueError(f"No PedestrianColor found with name {name}")