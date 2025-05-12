import torch
from ..common.tree import *

class IG_ImageCrop:
    """
    A node that crops an image by specified pixel amounts from the top, right, bottom, and left edges.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "min": 0, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop"
    CATEGORY = TREE_IO

    def crop(self, image: torch.Tensor, top: int, right: int, bottom: int, left: int):
        if image is None:
            return (None,)

        # Image tensor is expected in BCHW format (Batch, Channels, Height, Width)
        _b, h, w, _c = image.shape

        # Clamp crop values to ensure they are non-negative
        top = max(0, top)
        right = max(0, right)
        bottom = max(0, bottom)
        left = max(0, left)

        # Calculate crop boundaries
        y_start = top
        y_end = h - bottom
        x_start = left
        x_end = w - right

        # Validate crop dimensions to prevent slicing errors or zero-size tensors
        if y_start >= y_end or x_start >= x_end:
            raise ValueError(f"Invalid crop dimensions: Resulting image size would be non-positive ({x_end-x_start}x{y_end-y_start}). Original size: {w}x{h}, Crop: T{top}, R{right}, B{bottom}, L{left}")

        # Perform the crop using tensor slicing
        cropped_image = image[:, y_start:y_end, x_start:x_end, :]

        return (cropped_image,) 