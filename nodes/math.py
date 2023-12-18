import sys
from ..common.tree import *
from ..common.constants import *

class IG_MultiplyNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": FLOAT_STEP}),
                "Value_B": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": FLOAT_STEP}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = TREE_MATH

    def main(self, Value_A, Value_B):
        total = float(Value_A * Value_B)
        return (total,)