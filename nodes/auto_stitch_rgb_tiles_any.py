import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from ..common.tree import *            # 🐓 TREE_IO constant


class IG_AutoStitchRGBTilesAny:
    """
    Automatically stitches an unordered batch of horizontally-overlapping RGB
    tiles into a single image.  The node iteratively grows a 'combined' canvas:

        1. Pick one tile as the initial canvas.
        2. For every remaining tile, uses the ENTIRE candidate tile as
           cv2.matchTemplate template, sliding it over left & right bands
           (max_search px) of the current canvas.
        3. Select the tile/side with the *lowest* error below `max_err`.
           Blend it in with linear ramps, extend the canvas, repeat (2).
        4. Stop when no tile matches within `max_err`.

    Assumptions
    -----------
    • Tiles share the same height 𝐇  (widths may differ).
    • Overlaps are purely horizontal (no vertical shift / rotation).

    Inputs
    ------
    tiles        : IMAGE – [N,H,W,3] or [H,W,3] float 0-1
    min_overlap  : INT   – smallest overlap to test               (≥ 1)
    max_search   : INT   – largest overlap to test (0⇢auto)       (≤W)
    stride       : INT   – subsamples every `stride`-th pixel     (≥ 1)
    max_err      : FLOAT – maximum TM_SQDIFF_NORMED accepted      (default 0.1)
    debug        : BOOL  – print detailed progress messages

    Outputs
    -------
    stitched     : IMAGE – [1,H,W_out,3]
    order        : INT   – list of tile indices in the stitched order
    overlaps     : INT   – list of per-junction overlap widths
    """

    # ----------------------------------------------------------- #
    #  Tiny helper so we can sprinkle prints without clutter      #
    # ----------------------------------------------------------- #
    def _log(self, dbg: bool, msg: str):
        if dbg:
            print("[IG-Stitch-DBG]", msg)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "min_overlap": ("INT", {"default": 64,  "min": 1, "max": 99999999, "step": 1}),
                "max_search": ("INT", {"default": 0,   "min": 0, "max": 99999999, "step": 1}),
                "stride": ("INT",     {"default": 1,   "min": 1, "max": 99999999, "step": 1}),
                "max_err": ("FLOAT",  {"default": 0.1, "step": 0.01}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES   = ("IMAGE", "INT", "INT")
    RETURN_NAMES   = ("stitched", "order", "overlaps")
    FUNCTION       = "main"
    CATEGORY       = TREE_IO      # appears under 🐓 IG Nodes / IO

    # ----------------------------------------------------------- #
    #  Helpers                                                    #
    # ----------------------------------------------------------- #
    def _to_gray_u8(self, img: torch.Tensor) -> np.ndarray:
        """RGB float tensor [H,W,3] → uint8 grayscale on CPU."""
        img_u8 = (img.cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    def _slide_template(
        self,
        template: np.ndarray,
        search_band: np.ndarray,
    ) -> tuple[float, int]:
        """
        Run cv2.matchTemplate (TM_SQDIFF_NORMED) and return
        (best_score, best_shift).
        best_shift is the horizontal offset of the top-left corner of the
        template inside the search band.
        """
        res = cv2.matchTemplate(search_band, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        return float(min_val), int(min_loc[0])

    # ----------------------------------------------------------- #

    def main(
        self,
        tiles: torch.Tensor,
        min_overlap: int,
        max_search: int,
        stride: int,
        max_err: float,
        debug: bool = False,
    ):
        self._log(debug, f"Starting stitch: {tiles.shape[0]} tiles, min_overlap={min_overlap}, max_search={max_search}")

        # -------- normalise input -------------------------------- #
        if tiles.dim() == 3:
            tiles = tiles.unsqueeze(0)                  # [1,H,W,3]
        N, H, W0, C = tiles.shape
        if N == 1:
            return (tiles, [0], [])

        if max_search <= 0:
            max_search = max(t.shape[1] for t in tiles) - min_overlap

        # -------- init canvas with first tile -------------------- #
        order     : List[int] = [0]
        overlaps  : List[int] = []
        placed    : List[bool] = [False] * N
        placed[0] = True

        device, dtype = tiles.device, tiles.dtype
        stitched = tiles[0:1].clone()                    # [1,H,W,3]
        weight   = torch.ones_like(stitched)

        # pre-compute grayscale images for speed
        gray_imgs = [self._to_gray_u8(tiles[i, ...]) for i in range(N)]
        gray_canvas = gray_imgs[0]

        # -------- iterative placement ---------------------------- #
        while not all(placed):
            self._log(debug, f"--- iteration {len(order)} | canvas_w={stitched.shape[2]} ---")

            best_idx   = -1
            best_side  = None          # 'left' or 'right'
            best_err   = max_err
            best_shift = 0
            best_band_w = 0

            # =========================================================== #
            #  Search all remaining tiles                                 #
            # =========================================================== #
            canvas_left_band  = gray_canvas[:, :max_search]    # H × W_band
            canvas_right_band = gray_canvas[:, -max_search:]   # H × W_band

            for idx in range(N):
                if placed[idx]:
                    continue
                tpl = gray_imgs[idx]           # whole tile as template
                h, w_tpl = tpl.shape

                self._log(debug, f"  tile {idx:02d} | w={w_tpl} px")

                # Skip if template wider than band
                if w_tpl < min_overlap:
                    continue
                band_w = min(max_search, w_tpl)
                band_w = min(max_search, w_tpl, canvas_right_band.shape[1])

                # ---------- try placing tile to the RIGHT of canvas -----
                score, shift = self._slide_template(
                    template=tpl,
                    search_band=canvas_right_band[:, -band_w:],
                )
                self._log(debug, f"    RIGHT  score={score:.4f}  shift={shift}  band_w={band_w}")
                if score < best_err:
                    best_idx, best_side = idx, "right"
                    best_err, best_shift = score, shift
                    best_band_w = band_w

                # ---------- try placing tile to the LEFT of canvas ------
                score, shift = self._slide_template(
                    template=tpl,
                    search_band=canvas_left_band[:, :band_w],
                )
                self._log(debug, f"    LEFT   score={score:.4f}  shift={shift}  band_w={band_w}")
                if score < best_err:
                    best_idx, best_side = idx, "left"
                    best_err, best_shift = score, shift
                    best_band_w = band_w

            # -------------------------------------------------------------- #
            #  Determine overlap & new-columns for the chosen tile           #
            # -------------------------------------------------------------- #
            if best_idx == -1 or best_err > max_err:
                self._log(debug, "No candidate beat max_err — stopping.")
                break   # no suitable tile

            tile_w = tiles[best_idx].shape[2]
            overlap_px = best_band_w - best_shift
            new_cols   = tile_w - overlap_px
            if new_cols < min_overlap:
                self._log(debug, f"Best candidate would add only {new_cols} px — stopping.")
                break   # adds nothing useful

            self._log(debug, f"Chosen tile {best_idx} on {best_side} | err={best_err:.4f} | overlap={overlap_px} | new_cols={new_cols}")

            # -------- blend the chosen tile into canvas ---------- #
            tile = tiles[best_idx : best_idx + 1]   # keep batch dim

            if best_side == "right":
                # ------------------------------------------------------- #
                #  append tile on the RIGHT side of the canvas            #
                # ------------------------------------------------------- #
                offset = stitched.shape[2] - overlap_px          # where tile starts
                extra  = new_cols                     # new columns needed

                if extra > 0:                                 # grow canvas
                    zeros_img = torch.zeros(
                        (1, H, extra, C), dtype=dtype, device=device
                    )
                    stitched = torch.cat([stitched, zeros_img.clone()], dim=2)
                    weight   = torch.cat([weight,   zeros_img.clone()], dim=2)

                # blend with linear ramp
                ramp = torch.linspace(1.0, 0.0, overlap_px, device=device, dtype=dtype)
                mask_1d = torch.cat(
                    [torch.ones(tile_w - overlap_px, device=device, dtype=dtype), ramp]
                )
                mask = mask_1d.view(1, 1, tile_w, 1)          # broadcast → NHWC

                stitched[..., offset : offset + tile_w, :] += tile * mask
                weight  [..., offset : offset + tile_w, :] += mask

                order.append(best_idx)
                overlaps.append(overlap_px)

            else:
                # ------------------------------------------------------- #
                #  prepend tile on the LEFT side of the canvas            #
                # ------------------------------------------------------- #
                extra  = new_cols                     # new columns on left
                offset = extra                                # where old canvas shifts

                if extra > 0:                                 # grow canvas
                    zeros_img = torch.zeros(
                        (1, H, extra, C), dtype=dtype, device=device
                    )
                    stitched = torch.cat([zeros_img.clone(), stitched], dim=2)
                    weight   = torch.cat([zeros_img.clone(), weight  ], dim=2)

                # blend with linear ramp
                ramp = torch.linspace(0.0, 1.0, overlap_px, device=device, dtype=dtype)
                mask_1d = torch.cat(
                    [ramp, torch.ones(tile_w - overlap_px, device=device, dtype=dtype)]
                )
                mask = mask_1d.view(1, 1, tile_w, 1)

                stitched[..., :tile_w, :] += tile * mask
                weight  [..., :tile_w, :] += mask

                order.insert(0, best_idx)
                overlaps.insert(0, overlap_px)

            placed[best_idx] = True
            # keep a greyscale version of the stitched canvas for next iteration
            gray_canvas = self._to_gray_u8(stitched[0, ...])

        stitched /= torch.clamp_min(weight, 1e-8)
        self._log(debug, f"Finished.  final_w={stitched.shape[2]}  order={order}  overlaps={overlaps}")
        return (stitched, order, overlaps) 