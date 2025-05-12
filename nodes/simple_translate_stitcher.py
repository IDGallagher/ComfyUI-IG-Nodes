import cv2
import torch
import numpy as np
from ..common.tree import *                 # 🐓 TREE_IO


class IG_SimpleTranslateStitcher:
    """
    Greedy-canvas panorama stitcher (pure translation, no warp).

    Algorithm
    ---------
    1. Take the first tile as the canvas.
    2. Repeat until all tiles are merged:
       • For every remaining tile estimate its best horizontal offset
         **with respect to the *current* canvas**:
           – coarse x-offset via cv2.matchTemplate on small central bands
           – refine with cv2.findTransformECC (MOTION_TRANSLATION)
       • Pick the tile whose correlation coeff (cc) is highest and ≥ cc_threshold.
       • Paste that tile onto the canvas with linear alpha-blend over `blend_width_px`.
    3. If no candidate reaches the threshold the node raises RuntimeError.

    Assumptions
    -----------
    • All images share the same height.
    • Overlap is only horizontal (no rotation / scale).
    """

    # ---------- Comfy-UI interface ----------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":         ("IMAGE",),        # list input
                "blend_width_px": ("INT",   {"default": 64, "min": 0,  "max": 2000}),
                "max_iter":       ("INT",   {"default": 150, "min": 1, "max": 10000}),
                "eps_exponent":   ("INT",   {"default": -6, "min": -10, "max": -1}),
                "cc_threshold":   ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("stitched",)
    FUNCTION      = "main"
    CATEGORY      = TREE_IO
    # ----------------------------------------

    # ---------- helpers ----------
    @staticmethod
    def _to_gray(img_t: torch.Tensor) -> np.ndarray:
        if img_t.dim() == 4:
            img_t = img_t.squeeze(0)
        rgb = img_t.cpu().numpy()
        gray = (rgb[..., 0]*0.299 + rgb[..., 1]*0.587 + rgb[..., 2]*0.114).astype(np.float32)
        return cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX)

    @staticmethod
    def _to_rgb8(img_t: torch.Tensor) -> np.ndarray:
        if img_t.dim() == 4:
            img_t = img_t.squeeze(0)
        return (img_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # ---------- main ----------
    def main(self,
             images,
             blend_width_px     = 64,
             max_iter           = 150,
             eps_exponent       = -6,
             cc_threshold       = 0.80):

        # unwrap 1-element lists emitted by Comfy when INPUT_IS_LIST = True
        blend_width_px = blend_width_px[0] if isinstance(blend_width_px, list) else blend_width_px
        max_iter = max_iter[0] if isinstance(max_iter, list) else max_iter
        eps_exponent = eps_exponent[0] if isinstance(eps_exponent, list) else eps_exponent
        cc_threshold = cc_threshold[0] if isinstance(cc_threshold, list) else cc_threshold

        if not isinstance(images, (list, tuple)) or len(images) < 2:
            raise ValueError("Need an IMAGE_LIST containing at least 2 items.")

        # convert once
        gray_tiles = [self._to_gray(img) for img in images]
        rgb_tiles  = [self._to_rgb8(img) for img in images]

        # initial canvas = first tile
        canvas = rgb_tiles.pop(0)
        gray_canvas = gray_tiles.pop(0)
        offsets = [0]      # record where each original tile ended up (debug only)

        orb = cv2.ORB_create(2000)

        while rgb_tiles:
            best_idx   = None
            best_dx    = None
            best_cc    = -1.0
            best_tile  = None
            best_gray  = None

            # vertical strip (middle 60 %) speeds up matchTemplate & is robust to dark edges
            h_strip = slice(int(canvas.shape[0]*0.2), int(canvas.shape[0]*0.8))

            for idx, (g, tile_rgb) in enumerate(zip(gray_tiles, rgb_tiles)):
                # ------------------------------------------------------------
                # 1)   coarse search with template matching
                # ------------------------------------------------------------
                search = gray_canvas[h_strip, :]
                templ  = g[h_strip, :]
                res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
                _minVal, _maxVal, _minLoc, maxLoc = cv2.minMaxLoc(res)
                dx_init = maxLoc[0] - 0                          # horizontal shift guess

                # ------------------------------------------------------------
                # 2)   refine with ECC around the guessed region
                # ------------------------------------------------------------
                # Pad narrower image so ECC has same size arrays
                max_w = max(search.shape[1], templ.shape[1])
                pad = lambda arr: cv2.copyMakeBorder(arr, 0, 0, 0,
                                                     max_w - arr.shape[1],
                                                     cv2.BORDER_CONSTANT, value=0)
                src = pad(templ)
                dst = pad(search)

                warp = np.array([[1, 0, dx_init],
                                 [0, 1, 0]], dtype=np.float32)
                try:
                    cc, warp = cv2.findTransformECC(
                        src, dst, warp,
                        motionType=cv2.MOTION_TRANSLATION,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                  max_iter,
                                  10 ** eps_exponent),
                        gaussFiltSize=5)
                except cv2.error:
                    continue  # ECC failed – skip this candidate

                if cc > best_cc:
                    best_cc   = cc
                    best_idx  = idx
                    best_dx   = warp[0, 2]
                    best_tile = tile_rgb
                    best_gray = g

            # ---------- sanity check ----------
            if best_idx is None or best_cc < cc_threshold:
                raise RuntimeError(f"No remaining tile met cc_threshold={cc_threshold:.2f} "
                                   f"(best cc={best_cc:.2f}). Stitching aborted.")

            # remove chosen tile from lists
            rgb_tiles.pop(best_idx)
            gray_tiles.pop(best_idx)

            dx = int(round(best_dx))
            if dx < 0:
                # expand canvas on the left
                pad = np.zeros((canvas.shape[0], -dx, 3), dtype=np.uint8)
                canvas = np.hstack([pad, canvas])
                gray_canvas = np.hstack([np.zeros_like(pad[..., 0], dtype=np.float32), gray_canvas])
                offsets = [o - dx for o in offsets]   # shift previous offsets
                dx = 0

            # ensure canvas wide enough on the right
            new_end = dx + best_tile.shape[1]
            if new_end > canvas.shape[1]:
                pad = np.zeros((canvas.shape[0], new_end - canvas.shape[1], 3), dtype=np.uint8)
                canvas = np.hstack([canvas, pad])
                gray_canvas = np.hstack([gray_canvas,
                                         np.zeros_like(pad[..., 0], dtype=np.float32)])

            # ---------- paste with blend ----------
            bw = min(blend_width_px, best_tile.shape[1])
            # left non-overlap
            canvas[:, dx + bw:dx + best_tile.shape[1]] = best_tile[:, bw:]
            # blend region
            if bw > 0:
                alpha = np.linspace(0, 1, bw, endpoint=False, dtype=np.float32)[None, :, None]
                can_roi = canvas[:, dx:dx + bw].astype(np.float32)
                tile_roi = best_tile[:, :bw].astype(np.float32)
                canvas[:, dx:dx + bw] = (can_roi * (1 - alpha) + tile_roi * alpha).astype(np.uint8)

            # update gray canvas (for next round)
            gray_canvas[:, dx:dx + best_tile.shape[1]] = \
                (canvas[:, dx:dx + best_tile.shape[1]][..., ::-1].astype(np.float32) / 255.0
                 ).mean(axis=2)

            offsets.append(dx)

        stitched = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return (stitched,)
