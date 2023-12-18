import sys
import os
import json
import random
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional
import PIL.Image
import einops
from comfy.k_diffusion.utils import FolderOfImages

from ..common.tree import *
from ..common.constants import *

class IG_AnalyzeSSIM:
    
    def __init__(self) -> None:
        self.folder = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "ymin": ("FLOAT", {"default": 0}),
                "ymax": ("FLOAT", {"default": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = TREE_EXPLORER

    def main(self, folder, ymin, ymax):
        self.folder = folder
        ssim_file = os.path.join(folder, 'ssim_data.json')

        # Check if SSIM data already exists
        # if os.path.exists(ssim_file):
        if False:
            with open(ssim_file, 'r') as file:
                ssim_data = json.load(file)
        else:
            # Calculate SSIM and save it to JSON file
            ssim_data = self.calculate_ssim(folder)
            with open(ssim_file, 'w') as file:
                json.dump(ssim_data, file)

        # Plot the SSIM data
        image_tensor = self.plot_ssim_data(ssim_data, ymin, ymax)

        return (image_tensor, )
    
    def calculate_ssim(self, folder):
        files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in FolderOfImages.IMG_EXTENSIONS]
        ssim_values = []

        for i in range(len(files) - 1):
            file1 = os.path.join(folder, files[i])
            file2 = os.path.join(folder, files[i+1])
            print(f"File {file1} {file2}")
            img1 = imread(file1)
            img2 = imread(file2)
            # Convert the images to grayscale
            image1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ssim_val = ssim(image1_gray, image2_gray, multichannel=True)
            ssim_values.append(ssim_val)

        return ssim_values

    def plot_ssim_data(self, ssim_data, ymin, ymax):
        plt.figure(figsize=(12, 6))
        plt.plot(ssim_data)
        plt.title('SSIM Between Consecutive Images')
        plt.xlabel('Image Index')
        plt.ylabel('SSIM Value')

         # Set the range for the y-axis
        plt.ylim(ymin, ymax)

        plt.draw()
        plot_image = PIL.Image.frombytes('RGB', plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb())
        plt.close()

        plot_image = plot_image.resize((2048, 1024))
        image_tensor = torchvision.transforms.functional.to_tensor(plot_image)
        image_tensor = einops.rearrange(image_tensor, 'c h w -> h w c').unsqueeze(0)
        
        return image_tensor

