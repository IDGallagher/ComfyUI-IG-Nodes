import sys
from ..common.tree import *
from ..common.constants import *

class IG_Int:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "main"
    CATEGORY = TREE_PRIMITIVES

    def main(self, value):
        return (value,)
    
class IG_Float:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": FLOAT_STEP}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = TREE_PRIMITIVES

    def main(self, value):
        return (value,)
    
class IG_String:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING",{}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = TREE_PRIMITIVES

    def main(self, value):
        return (value,)
    
class IG_ZFill:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 1}),
                "fill": ("INT", {"default": 6, "min": 0, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = TREE_PRIMITIVES

    def main(self, value, fill):
        return (f"{value}".zfill(fill),)