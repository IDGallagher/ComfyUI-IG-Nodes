import cv2
import torch
import numpy as np

from ..common.tree import *                     # 🐓 TREE_IO


class IG_StitchImagesCV2:
    """
    Wraps OpenCV's high-level `Stitcher` class so you can feed a batch of
    partially-overlapping RGB tiles in *any* order and receive a stitched
    panorama.

    Inputs
    ------
    images : IMAGE  – Tensor [N,H,W,3] or [H,W,3] (0-1 float)
    mode   : ["PANORAMA", "SCANS"]
             PANORAMA (default) = camera rotates in place
             SCANS              = camera translates (e.g. flatbed scan)

    Outputs
    -------
    stitched : IMAGE – [1,H_out,W_out,3]  (0-1 float)

    Notes
    -----
    * Relies on OpenCV contrib (`opencv-contrib-python>=4.5`) because the
      Stitcher class lives in the contrib module.
    * If stitching fails (status != OK) the node raises a RuntimeError.
    """

    MODES = {
        "PANORAMA": cv2.Stitcher_PANORAMA,
        "SCANS":    cv2.Stitcher_SCANS,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode":   (list(cls.MODES.keys()),),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched",)
    FUNCTION = "main"
    CATEGORY = TREE_IO        # shows up under 🐓 IG Nodes / IO

    # --------------------------------------------------------- #
    def _tensor_to_bgr_u8(self, img: torch.Tensor) -> np.ndarray:
        """[H,W,3] 0-1 tensor ➜ uint8 BGR ndarray (CPU)."""
        img = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _bgr_to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """uint8 BGR ➜ float RGB tensor [1,H,W,3]."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0)          # add batch dim

    # --------------------------------------------------------- #
    def main(self, images: torch.Tensor, mode: str):
        # normalise dims to [N,H,W,3]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        imgs_bgr = [self._tensor_to_bgr_u8(im) for im in images]

        stitcher = cv2.Stitcher_create(self.MODES[mode])
        status, pano = stitcher.stitch(imgs_bgr)

        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"OpenCV Stitcher failed with status {status}")

        stitched_tensor = self._bgr_to_tensor(pano)
        return (stitched_tensor,) 