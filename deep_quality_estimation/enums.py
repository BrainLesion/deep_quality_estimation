from enum import Enum, auto


class View(Enum):
    """Enum for the different views of the brain"""

    AXIAL = auto()
    CORONAL = auto()
    SAGITTAL = auto()
