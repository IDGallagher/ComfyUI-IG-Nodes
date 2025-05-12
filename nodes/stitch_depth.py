# v1.1 – added wrap_horizontal for cylindrical panorama support (UE5.5)

import torch
from ..common.tree import *      # TREE_IO constant

class IG_StitchDepthTiles:
    """
    Re-assemble metric-depth tiles that were sliced horizontally with a fixed
    overlap.  When *wrap_horizontal=True* the first and last tiles are blended
    together so pixel 0 matches pixel W-1, producing a seamless cylindrical
    depth map.

    Inputs
    ------
    depth_tiles    : IMAGE  – [B × n_tiles, H, tile_width, C]
    tile_width     : INT    – Width of each tile (px)
    overlap_width  : INT    – Actual overlap used when tiling (px)
    image_width    : INT    – Original full image width (px)
    batch_size     : INT    – Number of original images in the batch
    wrap_horizontal: BOOL   – Enable cyclical blending (default False)

    Output
    ------
    depth : IMAGE – [B, H, image_width, C] stitched map(s)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_tiles": ("IMAGE",),
                "tile_width": ("INT",  {"default": 512, "min": 1}),
                "overlap_width": ("INT", {"default": 64, "min": 0}),
                "image_width": ("INT",),
                "batch_size": ("INT",  {"default": 1, "min": 1}),
                "wrap_horizontal": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("depth",)
    FUNCTION      = "main"
    CATEGORY      = TREE_IO

    # ------------------------------------------------------------------ #
    #  main                                                              #
    # ------------------------------------------------------------------ #
    def main(
        self,
        depth_tiles,
        tile_width: int,
        overlap_width: int,
        image_width: int,
        batch_size: int = 1,
        wrap_horizontal: bool = False,
    ):
        if depth_tiles.dim() != 4:
            raise ValueError("depth_tiles must be [B,H,W,C].")

        n_tiles = depth_tiles.shape[0] // batch_size
        if n_tiles * batch_size != depth_tiles.shape[0]:
            raise ValueError("batch_size does not divide depth_tiles batch dimension.")

        stride = tile_width - overlap_width
        H, C = depth_tiles.shape[1], depth_tiles.shape[3]

        # ----- reshape to [B, n_tiles, H, tile_width, C] --------------------
        depth_tiles = depth_tiles.view(batch_size, n_tiles, H, tile_width, C)

        device, dtype = depth_tiles.device, depth_tiles.dtype
        W_target = image_width

        out    = torch.zeros((batch_size, H, W_target, C), dtype=dtype, device=device)
        weight = torch.zeros_like(out)

        # ----- 1-D linear ramps for overlaps --------------------------------
        if overlap_width > 0:
            ramp_left  = torch.linspace(0.0, 1.0, overlap_width, device=device, dtype=dtype)
            ramp_right = torch.linspace(1.0, 0.0, overlap_width, device=device, dtype=dtype)

        for t in range(n_tiles):
            # ---------------- per-tile weight mask -------------------------
            mask_1d = torch.ones(tile_width, dtype=dtype, device=device)

            if overlap_width > 0:
                left_adjacent  = (t > 0) or wrap_horizontal
                right_adjacent = (t < n_tiles - 1) or wrap_horizontal

                if left_adjacent:
                    mask_1d[:overlap_width] = ramp_left
                if right_adjacent:
                    current = mask_1d[-overlap_width:]
                    mask_1d[-overlap_width:] = torch.minimum(current, ramp_right)

            mask = mask_1d.view(1, 1, tile_width, 1)   # broadcast over H & B
            depth_slice = depth_tiles[:, t] * mask

            # ---------------- paste into canvas ----------------------------
            start = (t * stride) % W_target if wrap_horizontal else t * stride
            end   = start + tile_width

            if wrap_horizontal and end > W_target:
                # split across boundary
                first_part  = W_target - start
                second_part = tile_width - first_part

                out[..., start:W_target, :]  += depth_slice[..., :first_part, :]
                weight[..., start:W_target, :] += mask[..., :first_part, :]

                out[..., 0:second_part, :]   += depth_slice[..., first_part:, :]
                weight[..., 0:second_part, :]  += mask[..., first_part:, :]
            else:
                if not wrap_horizontal and end > W_target:
                    start = W_target - tile_width   # final clamped tile
                    end   = W_target
                out[..., start:end, :]  += depth_slice
                weight[..., start:end, :] += mask

        depth = out / torch.clamp_min(weight, 1e-8)
        return (depth,)
