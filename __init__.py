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

NODE_CLASS_MAPPINGS = {
    "IG Multiply":          IG_MultiplyNode,   
    "IG Explorer":          IG_ExplorerNode,
    "IG Folder":            IG_Folder,       
    "IG Load Images":       IG_LoadImagesFromFolder,
    "IG Analyze SSIM":      IG_AnalyzeSSIM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG Multiply": "🧮 IG Multiply",
    "IG Explorer": "🤖 IG Explorer",
    "IG Folder": "📂 IG Folder",
    "IG Load Images": "📂 IG Load Images",
    "IG Analyze SSIM": "📉 Analyze SSIM",
}