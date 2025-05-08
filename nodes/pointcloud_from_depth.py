import math
import os
from typing import Tuple

import torch
import numpy as np

from ..common.tree import *   # supplies TREE_IO constant


class IG_PointCloudFromDepth:
    """
    Convert an RGB image + metric-depth map into a point-cloud tensor
    [N,6] = (x,y,z,r,g,b), suitable for writing to PLY or further processing.

    Geometry
    --------
    We treat the camera as pin-hole with intrinsics

        f_px       – focal length in **pixels**
        (c_x,c_y)  – principal point in pixels (origin at top-left).

    For a pixel (u,v) with metric depth d_r (radial distance from the camera
    centre) the 3-D **camera-space** coordinates are

        X = (u-c_x) * d_r / f_px
        Y = (v-c_y) * d_r / f_px
        Z =  d_r

    Inputs
    ------
    rgb_image        : IMAGE  – ComfyUI tensor [B,H,W,3] or [H,W,3]
    depth_image      : IMAGE  – Metric depth [same H,W] in metres
    focal_length_px  : FLOAT  – Focal length in pixels (default = W)
    principal_x_px   : INT    – If −1 ⇢ use image centre
    principal_y_px   : INT    – If −1 ⇢ use image centre
    stride           : INT    – >1 subsamples every N-th pixel to reduce N

    Outputs
    -------
    pointcloud : POINTCLOUD – torch.float32 [N,6] (x,y,z,r,g,b)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "depth_image": ("IMAGE",),
                "focal_length_px": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 99999999.0, "step": 1}),
                "principal_x_px": ("INT", {"default": -1, "min": -1, "max": 99999999, "step": 1}),
                "principal_y_px": ("INT", {"default": -1, "min": -1, "max": 99999999, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "main"
    CATEGORY = TREE_IO  # 🐓 IG Nodes/IO

    def main(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        focal_length_px: float,
        principal_x_px: int,
        principal_y_px: int,
        stride: int = 1,
    ):
        # Shape normalisation -------------------------------------------------
        if rgb_image.dim() == 3:
            rgb_image = rgb_image.unsqueeze(0)
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(0)

        if rgb_image.shape[0] != 1 or depth_image.shape[0] != 1:
            raise ValueError("Currently supports batch size 1 only.")
        if rgb_image.shape[1:3] != depth_image.shape[1:3]:
            raise ValueError("RGB and depth sizes must match.")

        B, H, W, _ = rgb_image.shape
        rgb = rgb_image[0]            # [H,W,3]
        depth = depth_image[0, :, :, 0]  # [H,W]

        # Camera intrinsics ---------------------------------------------------
        if focal_length_px <= 0:
            focal_length_px = float(W)  # sensible default: 1*image-width
        if principal_x_px < 0:
            principal_x_px = W * 0.5
        if principal_y_px < 0:
            principal_y_px = H * 0.5

        fx = fy = float(focal_length_px)
        cx = float(principal_x_px)
        cy = float(principal_y_px)

        # Sub-sampling mask ---------------------------------------------------
        v_idx = torch.arange(0, H, stride, device=depth.device)
        u_idx = torch.arange(0, W, stride, device=depth.device)
        vv, uu = torch.meshgrid(v_idx, u_idx, indexing="ij")  # [H_s,W_s]

        z = depth[vv, uu]                                     # [H_s,W_s]
        x = (uu.float() - cx) * z / fx
        y = (vv.float() - cy) * z / fy

        # Colours -------------------------------------------------------------
        rgb_s = rgb[vv, uu] * 255.0           # 0-1 → 0-255
        rgb_s = rgb_s.clamp(0, 255).to(torch.uint8)

        # Build Nx6 tensor ----------------------------------------------------
        xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(torch.float32)
        rgb_flat = rgb_s.reshape(-1, 3).to(torch.float32)

        pointcloud = torch.cat([xyz, rgb_flat], dim=1)  # [N,6]
        return (pointcloud,) 