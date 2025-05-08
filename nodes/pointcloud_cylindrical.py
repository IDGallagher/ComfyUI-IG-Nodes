import torch
import numpy as np
from ..common.tree import *   # supplies TREE_IO

class IG_PointCloudCylindricalFromDepth:
    """
    Convert an equirect-style 360° cylindrical panorama + metric-depth map
    into a point-cloud tensor [N, 6] = (x, y, z, r, g, b).

    Assumptions
    -----------
    • Horizontal FOV is fixed at 360° (left/right wrap).
    • Vertical angle ϕ is computed per-pixel from the given **focal_length_px**
      (same pin-hole model as the non-cylindrical node).  If you leave
      focal_length_px ≤ 0 we fall back to fx = W / (2π), which matches a
      unit-radius cylinder.
    • The depth image supplies **radial metric depth** (metres) measured
      from the camera focal point.

    Direction vector for pixel (u,v), 0-based top-left origin:

        θ =  2π · u / W  −  π              # yaw   ∈ [-π, π]
        φ =  atan2((v - cy), fy)           # pitch from focal length

        dir = (cosφ·sinθ ,  sinφ ,  cosφ·cosθ)

    The 3-D position is then  P = depth · dir.

    Inputs
    ------
    rgb_image   : IMAGE  – [H,W,3] or [1,H,W,3] float 0-1
    depth_image : IMAGE  – [H,W,1] or [1,H,W,1] metric depth (m)
    stride      : INT    – down-sample factor (>1 decimates pixels)

    Output
    ------
    pointcloud  : POINTCLOUD – torch.float32 [N,6] (x,y,z,r,g,b)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "depth_image": ("IMAGE",),
                "focal_length_px": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 99999999.0, "step": 1}),
                "principal_y_px": ("INT", {"default": -1, "min": -1, "max": 99999999, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "main"
    CATEGORY = TREE_IO           # appears under 🐓 IG Nodes/IO

    def main(self, rgb_image: torch.Tensor, depth_image: torch.Tensor, focal_length_px: float, principal_y_px: int, stride: int):
        # ------------------------------------------------------------------ #
        #  Input normalisation                                               #
        # ------------------------------------------------------------------ #
        if rgb_image.dim() == 3:
            rgb_image = rgb_image.unsqueeze(0)       # [1,H,W,3]
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(0)   # [1,H,W,1]

        if rgb_image.shape[0] != 1 or depth_image.shape[0] != 1:
            raise ValueError("Batch size >1 not supported yet.")
        if rgb_image.shape[1:3] != depth_image.shape[1:3]:
            raise ValueError("RGB & depth resolution must match.")

        _, H, W, _ = rgb_image.shape
        rgb  = rgb_image[0]              # [H,W,3] float 0-1
        depth = depth_image[0, :, :, 0]  # [H,W]   float metres

        # ------------------------------------------------------------------ #
        #  Build θ,φ grid (sub-sampled)                                      #
        # ------------------------------------------------------------------ #
        v_idx = torch.arange(0, H, stride, device=depth.device)
        u_idx = torch.arange(0, W, stride, device=depth.device)
        vv, uu = torch.meshgrid(v_idx, u_idx, indexing="ij")   # [H_s,W_s]

        # ------------------ camera intrinsics ----------------------- #
        if focal_length_px <= 0:                       # sensible default
            focal_length_px = W / (2 * torch.pi)       # fx ≈ radius of the pano
        fy = torch.tensor(focal_length_px, device=depth.device, dtype=depth.dtype)

        if principal_y_px < 0:
            cy = H * 0.5
        else:
            cy = float(principal_y_px)

        # ------------------------------------------------------------ #
        theta = (uu.float() / W) * (2 * torch.pi) - torch.pi           # [-π, π]
        # vertical angle: ϕ = arctan((v - cy)/fy)
        y_pix = vv.float() - cy
        phi   = torch.atan2(y_pix, fy)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # direction vector
        x = cos_phi * sin_theta
        y = sin_phi
        z = cos_phi * cos_theta

        d = depth[vv, uu]
        x *= d
        y *= d
        z *= d

        # ------------------------------------------------------------------ #
        #  Colour                                                            #
        # ------------------------------------------------------------------ #
        rgb_s = (rgb[vv, uu] * 255.0).clamp(0, 255).to(torch.uint8)

        # ------------------------------------------------------------------ #
        #  Stack to [N,6]                                                    #
        # ------------------------------------------------------------------ #
        xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(torch.float32)
        rgb_flat = rgb_s.reshape(-1, 3).to(torch.float32)

        pointcloud = torch.cat([xyz, rgb_flat], dim=1)  # [N,6]
        return (pointcloud,) 