import torch
import torch.nn.functional as F
from typing import List
from ..common.tree import *        # gives TREE_IO

class IG_AutoStitchRGBTiles:
    """
    Given a *batch* of overlapping RGB tiles that together form a single
    (wide) image, automatically discover the true horizontal overlap between
    neighbouring tiles and stitch them into one seamless picture.

    Assumptions
    -----------
    • Tiles are already ordered **left → right** in the batch.
    • All tiles share the same height H and width W.
    • Mis-alignments are horizontal only (no vertical shift, no rotation).

    Method
    ------
    For each consecutive pair of tiles (Tᵢ, Tᵢ₊₁) we search horizontal
    offsets `s ∈ [min_overlap, max_search]` and choose the *s* that minimises
    Mean-Squared-Error between

        Tᵢ[...,  W−s:W, :]   and   Tᵢ₊₁[..., 0:s, :]

    We then blend with linear ramps so seams vanish.  Each pair may discover
    a **different** overlap width, so ramps are generated per-pair.

    Inputs
    ------
    tiles        : IMAGE – tensor [N, H, W, 3] (float 0-1)
    min_overlap  : INT   – smallest overlap to test   (default 64)
    max_search   : INT   – largest overlap to test.  ≤0 → W-min_overlap
    stride       : INT   – >1 evaluates every *stride*-th pixel when
                           computing MSE (for speed).

    Outputs
    -------
    stitched     : IMAGE – [1, H, W_out, 3]                     (float)
    overlaps     : INT   – list[N-1] of discovered overlap px
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "min_overlap": ("INT",  {"default": 64,  "min": 1, "step": 1}),
                "max_search": ("INT",  {"default": 0,   "step": 1}),
                "stride": ("INT",      {"default": 1,   "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("stitched", "overlaps")
    FUNCTION = "main"
    CATEGORY = TREE_IO   #  🐓 IG Nodes / IO

    # --------------------------------------------------------------- #
    #  Main                                                           #
    # --------------------------------------------------------------- #
    def main(self,
             tiles: torch.Tensor,
             min_overlap: int,
             max_search: int,
             stride: int):

        # ---------- normalise input dims --------------------------- #
        if tiles.dim() == 3:
            tiles = tiles.unsqueeze(0)            # [1,H,W,3]
        if tiles.dim() != 4 or tiles.shape[-1] != 3:
            raise ValueError("tiles must be IMAGE tensor [N,H,W,3].")

        N, H, W, C = tiles.shape
        if N < 2:
            return (tiles, [])        # nothing to stitch

        if max_search <= 0 or max_search > W - min_overlap:
            max_search = W - min_overlap

        # ---------- utility: quickly compute MSE ------------------- #
        def mse(a, b):
            return torch.mean((a - b) ** 2)

        # ---------- discover overlap for each pair ----------------- #
        overlaps: List[int] = []
        for i in range(N - 1):
            left  = tiles[i,  ::stride, ::stride, :]
            right = tiles[i+1,::stride, ::stride, :]

            best_s      = min_overlap
            best_err    = float("inf")

            for s in range(min_overlap, max_search + 1):
                err = mse(left[:, -s:, :], right[:, :s, :])
                if err < best_err:
                    best_err, best_s = err.item(), s

            overlaps.append(best_s)

        # ---------- compute x-offsets ------------------------------- #
        offsets = [0]
        for s in overlaps:
            offsets.append(offsets[-1] + (W - s))
        W_out = offsets[-1] + W

        # ---------- allocate output & blend ramps ------------------ #
        device, dtype = tiles.device, tiles.dtype
        stitched = torch.zeros((1, H, W_out, C), dtype=dtype, device=device)
        weight   = torch.zeros_like(stitched)

        for idx in range(N):
            start = offsets[idx]
            end   = start + W
            mask_1d = torch.ones(W, dtype=dtype, device=device)

            # ramp on left edge (overlap with previous tile)
            if idx > 0:
                s_left = overlaps[idx-1]
                ramp = torch.linspace(0.0, 1.0, s_left, device=device, dtype=dtype)
                mask_1d[:s_left] = ramp

            # ramp on right edge (overlap with next tile)
            if idx < N - 1:
                s_right = overlaps[idx]
                ramp = torch.linspace(1.0, 0.0, s_right, device=device, dtype=dtype)
                curr = mask_1d[-s_right:]
                mask_1d[-s_right:] = torch.minimum(curr, ramp)

            mask = mask_1d.view(1, 1, W, 1)      # broadcast over H
            stitched[..., start:end, :] += tiles[idx] * mask
            weight  [..., start:end, :] += mask

        stitched /= torch.clamp_min(weight, 1e-8)
        return (stitched, overlaps) 