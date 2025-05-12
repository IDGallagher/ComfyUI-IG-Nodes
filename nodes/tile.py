# v1.3 – fixed wrap-mode off-by-one causing mismatched tile widths (UE5.5)

import math
import torch
from ..common.tree import *      # TREE_IO constant


class IG_TileImage:
    """
    Slice a wide image into fixed-width tiles while guaranteeing a minimum
    overlap between neighbours.

    New parameters
    --------------
    wrap_horizontal : bool
        When True the final tile wraps around the right edge so the first
        and last tiles overlap, creating edge-consistent input for
        cylindrical panoramas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width":        ("INT", {"default": 512, "min": 1}),
                "min_overlap_width": ("INT", {"default": 64,  "min": 0}),
                "wrap_horizontal":   ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("tiles", "overlap_width", "image_width")
    FUNCTION     = "main"
    CATEGORY     = TREE_IO

    # ------------------------------------------------------------------ #
    #  main                                                              #
    # ------------------------------------------------------------------ #
    def main(
        self,
        image: torch.Tensor,
        tile_width: int,
        min_overlap_width: int,
        wrap_horizontal: bool = False,
    ):
        # ---------- normalise rank ----------------------------------------
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError("IMAGE tensor must be [B,H,W,C] or [H,W,C].")
        B, H, W, C = image.shape

        # ---------- trivial case ------------------------------------------
        if W <= tile_width:
            return (image, 0, W)                # no tiling needed

        stride         = max(1, tile_width - min_overlap_width)
        overlap_width  = tile_width - stride
        tiles          = []

        # ---------- number of tiles ---------------------------------------
        if wrap_horizontal:
            n_tiles = math.ceil(W / stride)     # ensures last start < W
        else:
            n_tiles = math.ceil((W - tile_width) / stride) + 1

        # ---------- extract tiles -----------------------------------------
        for i in range(n_tiles):
            start = i * stride
            end   = start + tile_width

            if end <= W:
                tile = image[:, :, start:end, :]
            elif wrap_horizontal:               # split across boundary
                part1 = image[:, :, start:W, :]
                part2 = image[:, :, 0:end - W, :]
                tile  = torch.cat([part1, part2], dim=2)
            else:                               # clamp final non-wrap tile
                tile = image[:, :, W - tile_width:W, :]

            tiles.append(tile)

        tiles = torch.cat(tiles, dim=0)         # stack on batch dimension
        return (tiles, overlap_width, W)
