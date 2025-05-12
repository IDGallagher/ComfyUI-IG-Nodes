import cv2
import torch
import numpy as np

from ..common.tree import *                     # 🐓 TREE_IO


class IG_StitchImagesCV2:
    """
    Height-safe, *zero-warp* panorama stitcher built on OpenCV.

    • **SCANS** mode (translation-only) is forced by default – absolutely no
      cylindrical or spherical bending.
    • Warper type is selectable but starts on **plane** so the output height
      matches the input tiles.
    • A tiny centre-crop trims the occasional 1-pixel top/bottom sliver that
      OpenCV sometimes adds.
    • Uses ORB features (quick, rotation-invariant) and disables wave-correction
      to avoid secondary warping.
    • Confidence threshold controls minimum match quality (0-1).
    """

    MODES = {
        "SCANS":    cv2.Stitcher_SCANS,       # translation only
        "PANORAMA": cv2.Stitcher_PANORAMA,    # kept for completeness
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode":   (list(cls.MODES.keys()), {"default": "SCANS"}),
                "crop_to_input_height": ("BOOLEAN", {"default": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("stitched",)
    FUNCTION      = "main"
    CATEGORY      = TREE_IO

    # ---------- helpers ----------
    @staticmethod
    def _tensor_to_bgr_u8(img: torch.Tensor) -> np.ndarray:
        """[H,W,3] 0-1 tensor ➜ uint8 BGR ndarray (CPU)."""
        img = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
        """uint8 BGR ➜ float RGB tensor [1,H,W,3]."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0)          # add batch dim

    # ---------- main ----------
    def main(self,
             images: torch.Tensor,
             mode: str = "SCANS",
             crop_to_input_height: bool = True,
             confidence_threshold: float = 0.25):

        # When INPUT_IS_LIST is True Comfy-UI passes *all* inputs as 1-element
        # lists.  Unwrap the scalars here.
        mode = mode[0] if isinstance(mode, list) else mode
        crop_to_input_height = (
            crop_to_input_height[0]
            if isinstance(crop_to_input_height, list)
            else crop_to_input_height
        )
        confidence_threshold = (
            confidence_threshold[0]
            if isinstance(confidence_threshold, list)
            else confidence_threshold
        )

        if not isinstance(images, (list, tuple)):
            raise TypeError("Feed an IMAGE_LIST, not a single IMAGE/batch.")

        imgs_bgr = []
        for im in images:
            if im.dim() == 4:
                im = im.squeeze(0)
            if im.dim() != 3:
                raise ValueError("Each list item must be [H,W,3] or [1,H,W,3].")
            imgs_bgr.append(self._tensor_to_bgr_u8(im))

        base_h = imgs_bgr[0].shape[0]

        stitcher = cv2.Stitcher_create(self.MODES[mode])
        stitcher.setWaveCorrection(False)
        stitcher.setPanoConfidenceThresh(confidence_threshold)

        status, pano = stitcher.stitch(imgs_bgr)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"OpenCV Stitcher failed with status {status}")

        if crop_to_input_height and pano.shape[0] != base_h:
            y0 = (pano.shape[0] - base_h) // 2
            pano = pano[y0:y0 + base_h]

        return (self._bgr_to_tensor(pano),) 