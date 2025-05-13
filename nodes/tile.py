# v1.6 – simpler wrap logic: constant stride, deterministic tiles,
#        hard-cap on tile count to prevent OOM (UE5.5)

import math
import torch
from ..common.tree import *          # TREE_IO constant


class IG_TileImage:
    """
    Slice a wide image into fixed-width tiles.

    *wrap_horizontal=False* → classic sliding-window tiling with a
    uniform overlap.

    *wrap_horizontal=True*  → windows slide past the right edge and wrap
    from x=0; the final (wrapped) tile may have a **larger** overlap than
    its neighbours.  That is fine so long as every tile is extracted by
    the *same* deterministic rule, because any stage that needs the exact
    overlap can recompute it from (i, stride, tile_width, image_width).

    The node refuses to generate more than MAX_TILES to avoid runaway
    memory use; choose wider tiles or a larger overlap if that happens.
    """

    MAX_TILES = 2048                  # guardrail against OOM

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

    RETURN_TYPES  = ("IMAGE", "INT", "INT")
    RETURN_NAMES  = ("tiles", "overlap_width", "image_width")
    FUNCTION      = "main"
    CATEGORY      = TREE_IO

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
        # -------- ensure rank 4 -------------------------------------------
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError("IMAGE must be [B,H,W,C] or [H,W,C].")

        B, H, W, C = image.shape
        if W <= tile_width:                           # one-tile shortcut
            return (image, 0, W)

        # -------- stride & overlap (constant) -----------------------------
        stride = max(1, tile_width - min_overlap_width)
        overlap_width = tile_width - stride

        if wrap_horizontal:
            n_tiles = math.ceil(W / stride)
        else:
            n_tiles = math.ceil((W - tile_width) / stride) + 1

        if n_tiles > self.MAX_TILES:
            raise ValueError(
                f"Tiling would create {n_tiles} tiles (> {self.MAX_TILES}). "
                f"Increase 'tile_width', increase 'min_overlap_width', "
                f"or resize the image."
            )

        # -------- extract tiles -------------------------------------------
        tiles = []
        for i in range(n_tiles):
            start = (i * stride) % W
            end   = start + tile_width
            
            print(f"Tile {i+1}/{n_tiles}:")
            print(f"  start: {start}")
            print(f"  end: {end}")
            print(f"  stride: {stride}")
            print(f"  tile_width: {tile_width}")

            if end <= W:
                print(f"  Taking slice [{start}:{end}]")
                tile = image[:, :, start:end, :]
            else:                                       # wrap split
                print(f"  Wrapping around image width {W}")
                right = image[:, :, start:W, :]
                done = W-start
                print(f"  Right portion: [{start}:{W}] ({done} pixels)")
                left  = image[:, :, :tile_width-done, :]
                print(f"  Left portion: [0:{tile_width-done}] ({tile_width-done} pixels)")
                tile  = torch.cat([right, left], dim=2)
                print(f"  Combined tile width: {tile.shape[2]}")

            tiles.append(tile)

        tiles = torch.cat(tiles, dim=0)                 # [B×T, H, Wt, C]
        print(f"\nFinal output shape: {tiles.shape}")
        return (tiles, overlap_width, W)
