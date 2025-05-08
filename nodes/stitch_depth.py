import torch
from ..common.tree import *           # provides TREE_IO constant

class IG_StitchDepthTiles:
    """
    Re-assemble metric-depth tiles which were sliced horizontally with
    at least `overlap_width` pixels shared between neighbours.

    Inputs
    ------
    depth_tiles    : IMAGE  – Tensor [B*n_tiles, H, tile_width, C]
                                (output of Depth Pro on the image tiles)
    tile_width     : INT    – Width (px) of each tile (same as used when slicing).
    overlap_width  : INT    – Actual overlap (px) between successive tiles.
    image_width    : INT    – Original width of the input image (pixels).
                             Output width will match this exactly.
    batch_size     : INT    – #original images packed in this batch
                              (default = 1; must divide depth_tiles.shape[0]).

    Outputs
    -------
    depth          : IMAGE  – Stitched metric-depth map(s)
                              [B, H, W_full, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_tiles": ("IMAGE",),
                "tile_width": ("INT", {"min": 1, "default": 512, "step": 1}),
                "overlap_width": ("INT", {"min": 0, "default": 64, "step": 1}),
                "image_width": ("INT",),   # original W from the tiler
                "batch_size": ("INT", {"min": 1, "default": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "main"
    CATEGORY = TREE_IO    # appears under 🐓 IG Nodes/IO

    def main(
        self,
        depth_tiles,
        tile_width: int,
        overlap_width: int,
        image_width: int,
        batch_size: int = 1,
    ):
        if depth_tiles.dim() != 4:
            raise ValueError("depth_tiles must be a 4-D IMAGE tensor [B, H, W, C].")

        M, H, Wt, C = depth_tiles.shape
        if Wt != tile_width:
            raise ValueError("Provided tile_width does not match tensor width.")
        if overlap_width < 0 or overlap_width >= tile_width:
            raise ValueError("overlap_width must be in [0, tile_width-1].")
        if M % batch_size != 0:
            raise ValueError("batch_size must divide the number of tiles.")

        n_tiles = M // batch_size
        stride = tile_width - overlap_width

        # Reshape to [B, n_tiles, H, tile_width, C]
        depth_tiles = depth_tiles.view(batch_size, n_tiles, H, tile_width, C)

        device, dtype = depth_tiles.device, depth_tiles.dtype
        W_target = image_width
        out = torch.zeros((batch_size, H, W_target, C), dtype=dtype, device=device)
        weight = torch.zeros_like(out)

        if overlap_width == 0:
            ramp_left = ramp_right = None  # not used
        else:
            ramp_left  = torch.linspace(0.0, 1.0, overlap_width, device=device, dtype=dtype)
            ramp_right = torch.linspace(1.0, 0.0, overlap_width, device=device, dtype=dtype)

        for t in range(n_tiles):
            # NOTE:
            #   Using torch.minimum() ensures that if three (or more) tiles overlap,
            #   the central tile does not suddenly regain full weight where the two
            #   ramps intersect – a common source of vertical seams.

            # base weight = 1 everywhere
            mask_1d = torch.ones(tile_width, device=device, dtype=dtype)

            if overlap_width:
                # Apply the ramps _without overwriting_.  
                # We always keep the **smallest** weight where two ramps overlap,
                # which yields a neat triangular profile even when the overlap
                # consumes more than half the tile width (i.e. 3-way blends).
                if t > 0:                              # left side of tile
                    mask_1d[:overlap_width] = ramp_left

                if t < n_tiles - 1:                    # right side of tile
                    current = mask_1d[-overlap_width:]
                    mask_1d[-overlap_width:] = torch.minimum(current, ramp_right)

            mask = mask_1d.view(1, 1, tile_width, 1)          # broadcast over H & B

            start = t * stride
            if start + tile_width > W_target:    # only triggers on the last tile
                start = W_target - tile_width
            end = start + tile_width
            depth_slice = depth_tiles[:, t] * mask

            out[..., start:end, :] += depth_slice
            weight[..., start:end, :] += mask

        depth = out / torch.clamp_min(weight, 1e-8)           # avoid divide-by-zero
        return (depth,) 