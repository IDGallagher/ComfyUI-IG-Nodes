import os
import numpy as np
import torch
from datetime import datetime
import folder_paths
from ..common.tree import *        # TREE_IO

class IG_SavePLYPointCloud:
    """
    Save a point-cloud tensor [N, 6] → (x, y, z, r, g, b) to disk.

    Supported formats
    -----------------
    • *ply_binary* – binary little-endian PLY (previous behaviour)  
    • *xyz_ascii*  – plain-text XYZRGB: one line per point  
                     "x y z r g b" (floats then ints)

    Inputs
    ------
    pointcloud      : POINTCLOUD   – torch float/uint8 [N,6]
    filename_prefix : STRING       – optional; auto-generated if empty
    export_format   : ["ply_binary", "xyz_ascii"]

    Output
    ------
    file_path       : STRING       – full path of the saved file
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pointcloud": ("POINTCLOUD",),
                "filename_prefix": ("STRING", {"default": ""}),
                "export_format": (["ply_binary", "xyz_ascii"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "main"
    CATEGORY = TREE_IO  # 🐓 IG Nodes / IO

    def main(self, pointcloud: torch.Tensor, filename_prefix: str,
             export_format: str):

        if pointcloud.dim() != 2 or pointcloud.shape[1] != 6:
            raise ValueError("pointcloud must be [N,6] (x,y,z,r,g,b).")

        # ------------------------------------------------------------------ #
        #  Output filename / folder                                          #
        # ------------------------------------------------------------------ #
        if not filename_prefix:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"pointcloud_{ts}"

        (full_output_folder, filename, counter, _sub, _prefix
         ) = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory())

        if export_format == "xyz_ascii":
            file = f"{filename}_{counter:05}_.xyz"
            file_path = os.path.join(full_output_folder, file)
            self._save_xyz_ascii(pointcloud, file_path)
            print(f"Saving XYZ file to {file_path}")
            return (file_path,)

        # ---------- default PLY-binary path ----------
        file = f"{filename}_{counter:05}_.ply"
        file_path = os.path.join(full_output_folder, file)
        self._save_ply_binary(pointcloud, file_path)
        print(f"Saving PLY file to {file_path}")
        return (file_path,)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_xyz_ascii(pc: torch.Tensor, path: str):
        """
        Write plain ASCII:
            x y z r g b\n
        with RGB as 0-255 integers.
        """
        pc_np = pc.cpu().numpy()
        xyz = pc_np[:, :3]
        rgb = np.clip(pc_np[:, 3:], 0, 255).astype(np.uint8)

        with open(path, "w", encoding="utf-8") as f:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    @staticmethod
    def _save_ply_binary(pc: torch.Tensor, path: str):
        """
        Original binary-little-endian PLY writer (unchanged from prior
        implementation).
        """
        pc_np = pc.cpu().numpy()
        xyz = pc_np[:, :3].astype("<f4")
        rgb = np.clip(pc_np[:, 3:], 0, 255).astype(np.uint8)

        N = pc_np.shape[0]
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")

        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")
        ])
        vertices = np.empty(N, dtype=vertex_dtype)
        vertices["x"], vertices["y"], vertices["z"] = xyz.T
        vertices["red"], vertices["green"], vertices["blue"] = rgb.T

        with open(path, "wb") as f:
            f.write(header)
            vertices.tofile(f) 