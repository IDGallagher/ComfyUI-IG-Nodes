import math
import torch

from ..common.tree import *  # brings in TREE_IO constant

class IG_TileImage:
    """
    Slice a wide image into fixed-width tiles (left-to-right) while guaranteeing
    a minimum overlap between neighbouring tiles.

    Inputs
    ------
    image : IMAGE           – ComfyUI tensor [B,H,W,C] or [H,W,C]
    tile_width : INT        – Desired tile width in pixels.
    min_overlap_width : INT – Minimum overlap (pixels) each tile must share
                              with its neighbour.

    Outputs
    -------
    tiles         : IMAGE – Batch of tiles ordered left→right.  Shape:
                            [B*n_tiles, H, tile_width, C]
    overlap_width : INT   – The actual overlap assigned (≥ min_overlap_width).
    image_width   : INT   – Original width of the input image (pixels).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 99999999, "step": 1}),
                "min_overlap_width": ("INT", {"default": 64, "min": 0, "max": 99999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("tiles", "overlap_width", "image_width")
    FUNCTION = "main"
    CATEGORY = TREE_IO     # Appears in 🐓 IG Nodes/IO

    def main(self, image, tile_width: int, min_overlap_width: int):
        # Accept either [H,W,C] or [B,H,W,C]
        if image.dim() == 3:         # single image, no batch dim
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError("IMAGE tensor must have 3 or 4 dimensions (B,H,W,C).")

        B, H, W, C = image.shape

        # If the image already fits in one tile, return it unchanged
        if W <= tile_width:
            return (image, 0, W)

        # Determine smallest stride that still meets the minimum overlap
        min_stride = max(1, tile_width - min_overlap_width)

        # Calculate number of tiles required
        n_tiles = math.ceil((W - tile_width) / min_stride) + 1

        # Compute the real overlap so that tiles exactly span the width
        overlap_width = 0
        if n_tiles > 1:
            overlap_width = int(round((n_tiles * tile_width - W) / (n_tiles - 1)))

        stride = tile_width - overlap_width

        # Extract tiles
        tiles = []
        for t in range(n_tiles):
            start = t * stride
            end   = start + tile_width
            if end > W:                # final tile: clamp to the right edge
                start = W - tile_width
                end   = W
            tile = image[:, :, start:end, :]   # keep original batch dim
            tiles.append(tile)

        tiles = torch.cat(tiles, dim=0)        # stack along batch dimension
        return (tiles, overlap_width, W) 