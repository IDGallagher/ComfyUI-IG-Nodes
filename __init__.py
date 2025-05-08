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
from .nodes.sm import *
from .nodes.tile import *
from .nodes.stitch_depth import *
from .nodes.pointcloud_from_depth import *
from .nodes.ply_export import *
from .nodes.pointcloud_cylindrical import *
from .nodes.auto_stitch_rgb_tiles import *
from .nodes.auto_stitch_rgb_tiles_any import *

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
    "SM Video Base":        SM_VideoBase,
    "SM Video Base Control": SM_VideoBaseControl,
    "IG Tile Image":        IG_TileImage,
    "IG Stitch Depth Tiles": IG_StitchDepthTiles,
    "IG PointCloud From Depth": IG_PointCloudFromDepth,
    "IG Save PLY PointCloud":   IG_SavePLYPointCloud,
    "IG PointCloud From Cylindrical": IG_PointCloudCylindricalFromDepth,
    "IG Auto Stitch RGB Tiles": IG_AutoStitchRGBTiles,
    "IG Auto Stitch RGB Tiles Any": IG_AutoStitchRGBTilesAny,
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
    "SM Video Base":        "🎞️ SM Video Base",
    "SM Video Base Control": "🎞️ SM Video Base Control",
    "IG Tile Image":        "🧩 IG Tile Image",
    "IG Stitch Depth Tiles": "🧩 IG Stitch Depth Tiles",
    "IG PointCloud From Depth": "🌐 IG PointCloud From Depth",
    "IG Save PLY PointCloud":   "💾 IG Save PLY PointCloud",
    "IG PointCloud From Cylindrical": "🌐 IG PointCloud From Cylindrical",
    "IG Auto Stitch RGB Tiles": "🧩 IG Auto Stitch RGB Tiles",
    "IG Auto Stitch RGB Tiles Any": "🧩 IG Auto Stitch RGB Tiles (Any)",
}