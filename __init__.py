"""
@author: IDGallagher
@title: IG Interpolation Nodes
@nickname: IG Interpolation Nodes
@description: Custom nodes to aid in the exploration of Latent Space
"""

from .nodes.math import *
from .nodes.explorer import *
from .nodes.io import *
from .nodes.analyze import *
from .nodes.primitives import *
from .nodes.interpolate import *

NODE_CLASS_MAPPINGS = {
    "IG Multiply":          IG_MultiplyNode,   
    "IG Explorer":          IG_ExplorerNode,
    "IG Folder":            IG_Folder,      
    "IG Load Image":        IG_LoadImage, 
    "IG Load Images":       IG_LoadImagesFromFolder,
    "IG Analyze SSIM":      IG_AnalyzeSSIM,
    "IG Int":               IG_Int,
    "IG Float":             IG_Float,
    "IG String":            IG_String,
    "IG Path Join":         IG_PathJoin,
    "IG Cross Fade Images": IG_CrossFadeImages,
    "IG Interpolate":       IG_Interpolate,
    "IG MotionPredictor":   IG_MotionPredictor,
    "IG ZFill":             IG_ZFill,
    "IG String List":       IG_StringList,
    "IG Float List":        IG_FloatList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG Multiply":          "🧮 IG Multiply",
    "IG Explorer":          "🤖 IG Explorer",
    "IG Folder":            "📂 IG Folder",
    "IG Load Image":        "📂 IG Load Image",
    "IG Load Images":       "📂 IG Load Images",
    "IG Analyze SSIM":      "📉 Analyze SSIM",
    "IG Int":               "➡️ IG Int",
    "IG Float":             "➡️ IG Float",
    "IG String":            "➡️ IG String",
    "IG Path Join":         "📂 IG Path Join",
    "IG Cross Fade Images": "🧑🏻‍🧑🏿‍🧒🏽 IG Cross Fade Images",
    "IG Interpolate":       "🧑🏻‍🧑🏿‍🧒🏽 IG Interpolate",
    "IG MotionPredictor":   "🏃‍♀️ IG Motion Predictor",
    "IG ZFill":             "⌨️ IG ZFill",
    "IG String List":       "📃 IG String List",
    "IG Float List":        "📃 IG Float List",
}