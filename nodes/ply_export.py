import os
import numpy as np
import torch
from datetime import datetime
import folder_paths
from ..common.tree import *        # TREE_IO

class IG_SavePLYPointCloud:
    """
    Write a point-cloud tensor [N,6] = (x,y,z,r,g,b) to a binary-little-endian
    .ply file.  Returns the file path for downstream use.

    Inputs
    ------
    pointcloud : POINTCLOUD – torch.float32 / float64 [N,6]
    filename   : STRING      – optional.  If empty → auto-generate.

    Output
    ------
    ply_path   : STRING      – path to the written file
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pointcloud": ("POINTCLOUD",),
                "filename_prefix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "main"
    CATEGORY = TREE_IO  # 🐓 IG Nodes/IO

    def main(self, pointcloud: torch.Tensor, filename_prefix: str):
        if pointcloud.dim() != 2 or pointcloud.shape[1] != 6:
            raise ValueError("pointcloud must be [N,6] (x,y,z,r,g,b).")

        pc_np = pointcloud.cpu().numpy()
        xyz = pc_np[:, :3].astype("<f4")       # little-endian float32
        rgb = pc_np[:, 3:].astype(np.uint8)    # 0-255

        N = pc_np.shape[0]

        # ------------------------------------------------------------------ #
        #  Build header                                                      #
        # ------------------------------------------------------------------ #
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
        header_bytes = ("\n".join(header)).encode("ascii")

        # ------------------------------------------------------------------ #
        #  Data block                                                        #
        # ------------------------------------------------------------------ #
        vertex_dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )
        vertices = np.empty(N, dtype=vertex_dtype)
        vertices["x"] = xyz[:, 0]
        vertices["y"] = xyz[:, 1]
        vertices["z"] = xyz[:, 2]
        vertices["red"] = rgb[:, 0]
        vertices["green"] = rgb[:, 1]
        vertices["blue"] = rgb[:, 2]

        # ------------------------------------------------------------------ #
        #  Output path                                                       #
        # ------------------------------------------------------------------ #
        if not filename_prefix:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"pointcloud_{ts}"

        if filename_prefix.endswith(".ply"):
            filename_prefix = filename_prefix.split(".ply")[0]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())

        file = f"{filename}_{counter:05}_.ply"
        ply_path = os.path.join(full_output_folder, file)

        print(f"Saving PLY file to {ply_path}")

        with open(ply_path, "wb") as f:
            f.write(header_bytes)
            vertices.tofile(f)

        return (ply_path,) 