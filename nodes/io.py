import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

from comfy.k_diffusion.utils import FolderOfImages

import sys
import folder_paths
from ..common.tree import *
from ..common.constants import *
from ..common.utils import calculate_file_hash, get_sorted_dir_files_from_directory


def is_changed_load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
    if not os.path.isdir(directory):
            return False
        
    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
    dir_files = dir_files[:image_load_cap]

    m = hashlib.sha256()
    for filepath in dir_files:
        m.update(calculate_file_hash(filepath).encode()) # strings must be encoded before hashing
    return m.digest().hex()

def is_changed_image(filepath: str):
    m = hashlib.sha256()
    m.update(calculate_file_hash(filepath).encode()) # strings must be encoded before hashing
    return m.digest().hex()

def validate_load_images(directory: str, **kwargs):
    if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
        return f"No files in directory '{directory}'."

    return True


def load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
    directory = folder_paths.get_annotated_filepath(directory.strip())
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory} cannot be found.")

    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")

    images = []
    masks = []

    limit_images = False
    if image_load_cap > 0:
        limit_images = True
    image_count = 0

    for image_path in dir_files:
        if limit_images and image_count >= image_load_cap:
            break
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        images.append(image)
        masks.append(mask)
        image_count += 1
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

    return (torch.cat(images, dim=0), torch.stack(masks, dim=0), image_count)

class IG_LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "main"

    CATEGORY = TREE_IO

    def main(self, image_path: str, **kwargs):
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    
    @classmethod
    def IS_CHANGED(s, image_path: str, **kwargs):
        return is_changed_image(image_path, **kwargs)

class IG_LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "main"

    CATEGORY = TREE_IO

    def main(self, folder: str, **kwargs):
        return load_images(folder, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, folder: str, **kwargs):
        return is_changed_load_images(folder, **kwargs)

    # @classmethod
    # def VALIDATE_INPUTS(s, folder: str, **kwargs):
    #     return validate_load_images(folder, **kwargs)


class IG_Folder:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        folder_type = ["input folder", "output folder"]
        return {
            "required": {
                "folder_parent": (folder_type, ),
                "folder_name": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = TREE_IO

    def main(self, folder_parent, folder_name):
        parent = folder_paths.input_directory if folder_parent == "folder_parent" else folder_paths.output_directory
        directory = os.path.join(parent, folder_name)
        return (directory,)
    
class IG_PathJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("STRING", {"default": '', "multiline": False}),
                "second": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = TREE_IO
    def main(self, first, second):
        path = os.path.join(first, second)
        return (path,)