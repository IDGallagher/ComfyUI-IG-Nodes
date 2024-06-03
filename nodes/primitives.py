import sys
import re
from copy import deepcopy
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
    
class IG_FloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "decimal_places": ("INT", {"default": 3, "min": 1}),
                "float_list": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("FLOAT","FLOAT","STRING_LIST")
    RETURN_NAMES = ("current value", "list", "list text")
    FUNCTION = "ParseAndReturnFloat"
    CATEGORY = TREE_PRIMITIVES

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def ParseAndReturnFloat( self, index: int, decimal_places: int, float_list: str ) -> tuple[float, list[float], list[str]]:
        if float_list != self.static_text:
            split_str = re.split( ",|;|\s|:", float_list )
            out_arr = []
            for val in split_str:
                # let the exception happen if invalid
                out_arr.append(float(val))
            self.static_text = float_list
            self.static_out_arr = deepcopy( out_arr )
        ret_val = 0.0
        if ( index > len(self.static_out_arr) ):
            ret_val = self.static_out_arr[-1]
        else:
            ret_val = self.static_out_arr[ index - 1 ]
        return (ret_val,self.static_out_arr, ["{:.{}f}".format(val, decimal_places) for val in self.static_out_arr])

class IG_StringList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "string_list": ("STRING", {"multiline": True}), 
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("current value","list")
    FUNCTION = "SplitAndReturnStrings"
    CATEGORY = TREE_PRIMITIVES

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def SplitAndReturnStrings( self, index: int, string_list: str ) -> tuple[str, list[str]]:
        if string_list != self.static_text:
            # unlike the numeric list nodes, we only want to split on newlines
            # TODO: support manual delimiter specification?
            split_str = re.split( "\n", string_list )
            self.static_text = string_list
            self.static_out_arr = deepcopy( split_str )
        if ( index > len(self.static_out_arr) ):
            return ( self.static_out_arr[len(self.static_out_arr) - 1], self.static_out_arr)
        return (self.static_out_arr[ index - 1 ], self.static_out_arr)