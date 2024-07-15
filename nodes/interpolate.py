import os
import hashlib
import numpy as np
import torch
import math
import json

import sys
from comfy import model_management
import folder_paths
from ..common.tree import *
from ..common.constants import *
from ..motion_predictor import MotionPredictor
import comfy.utils

def crossfade(images_1, images_2, alpha):
    crossfade = (1 - alpha) * images_1 + alpha * images_2
    return crossfade
def ease_in(t):
    return t * t
def ease_out(t):
    return 1 - (1 - t) * (1 - t)
def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t
def bounce(t):
    if t < 0.5:
        return ease_out(t * 2) * 0.5
    else:
        return ease_in((t - 0.5) * 2) * 0.5 + 0.5
def elastic(t):
    return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))
def glitchy(t):
    return t + 0.1 * math.sin(40 * t)
def exponential_ease_out(t):
    return 1 - (1 - t) ** 4

easing_functions = {
    "linear": lambda t: t,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
    "elastic": elastic,
    "glitchy": glitchy,
    "exponential_ease_out": exponential_ease_out,
}

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

class IG_MotionPredictor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embeds": ("PROJ_EMBEDS",),
                "neg_embeds": ("PROJ_EMBEDS",),
                "transitioning_frames": ("INT", {"default": 16,"min": 0, "max": 4096, "step": 1}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "mode": (["motion_predict", "interpolate_linear"], ),
                "motion_predictor_file": (folder_paths.get_filename_list("ipadapter"),),
            }, 
            "optional": {
                "positive_prompts": ("STRING", {"default": [], "forceInput": True}),
                "negative_prompts": ("STRING", {"default": [], "forceInput": True}),
            }
        }

    RETURN_TYPES = ("PROJ_EMBEDS", "PROJ_EMBEDS", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("pos_embeds", "neg_embeds", "positive_string", "negative_string", "BATCH_SIZE", )
    FUNCTION = "main"
    CATEGORY = TREE_INTERP 

    @torch.inference_mode()
    def main(self, pos_embeds, neg_embeds, transitioning_frames, repeat_count, mode, motion_predictor_file, positive_prompts=None, negative_prompts=None):
        
        torch_device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
    
        easing_function = easing_functions["linear"]
        
        print( f"Embed shape {pos_embeds.shape}")
        
        inbetween_embeds = []
        # Make sure we have 2 images
        if len(pos_embeds) > 1:
            if mode == "motion_predict":
                motion_predictor = MotionPredictor(total_frames=transitioning_frames).to(torch_device, dtype=dtype)
                motion_predictor_path = folder_paths.get_full_path("ipadapter", motion_predictor_file)
                checkpoint = comfy.utils.load_torch_file(motion_predictor_path, safe_load=True)
                motion_predictor.load_state_dict(checkpoint)
                for i in range(len(pos_embeds) - 1):
                    embed1 = pos_embeds[i]
                    embed2 = pos_embeds[i + 1]
                    embed1 = embed1.unsqueeze(0)
                    embed2 = embed2.unsqueeze(0)
                    inbetween_embeds = motion_predictor(embed1, embed2).squeeze(0)
            elif mode == "interpolate_linear":
                # Interpolate embeds
                for i in range(len(pos_embeds) - 1):
                    embed1 = pos_embeds[i]
                    embed2 = pos_embeds[i + 1]
                    alphas = torch.linspace(0, 1, transitioning_frames)
                    for alpha in alphas:
                        eased_alpha = easing_function(alpha.item())
                        print(f"eased alpha {eased_alpha}")
                        inbetween_embed = (1 - eased_alpha) * embed1 + eased_alpha * embed2
                        inbetween_embeds.extend([inbetween_embed])
                        
            inbetween_embeds = [embed for embed in inbetween_embeds for _ in range(repeat_count)]
            # Find size of batch
            batch_size = len(inbetween_embeds)

        inbetween_embeds = torch.stack(inbetween_embeds, dim=0)

        # ensure that cond and uncond have the same batch size
        neg_embeds = tensor_to_size(neg_embeds, inbetween_embeds.shape[0])

        # Combine and format prompt strings
        def format_text_prompts(text_prompts):
            string = ""
            for i, prompt in enumerate(text_prompts):
                string += f"\"{i * transitioning_frames * repeat_count - 1}\":\"{prompt}\",\n"
            return string
        
        positive_string = format_text_prompts(positive_prompts) if positive_prompts is not None and len(positive_prompts) > 0 else "\"0\":\"\",\n"
        negative_string = format_text_prompts(negative_prompts) if negative_prompts is not None and len(negative_prompts) > 0 else "\"0\":\"\",\n"
        
        return (inbetween_embeds, neg_embeds, positive_string, negative_string, batch_size,)

class IG_Interpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                "buffer": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }, 
            "optional": {
                "input_images1": ("IMAGE",),
                "input_images2": ("IMAGE",),
                "input_images3": ("IMAGE",),
                "positive_prompts": ("STRING", {"default": [], "forceInput": True}),
                "negative_prompts": ("STRING", {"default": [], "forceInput": True}),
            }
        }

    RETURN_TYPES = ("EMBEDS", "EMBEDS", "EMBEDS", "EMBEDS", "STRING", "STRING", "IPADAPTER", "INT", "STRING",)
    RETURN_NAMES = ("pos_embeds1", "pos_embeds2", "pos_embeds3", "neg_embeds", "positive_string", "negative_string", "ipadapter", "BATCH_SIZE", "FRAMES_TO_DROP",)
    FUNCTION = "main"
    CATEGORY = TREE_INTERP 

    @torch.inference_mode()
    def main(self, ipadapter, clip_vision, transitioning_frames, repeat_count, interpolation, buffer, input_images1=None, input_images2=None, input_images3=None, positive_prompts=None, negative_prompts=None):
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision
        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")
        
        is_plus = "proj.3.weight" in ipadapter_model["image_proj"] or "latents" in ipadapter_model["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]

        easing_function = easing_functions[interpolation]

        input = [input_images1, input_images2, input_images3]
        output = []
        for input_images in input:
            if input_images == None:
                continue
            # Create pos embeds
            img_cond_embeds = clip_vision.encode_image(input_images)
            print( f"penultimate_hidden_states shape {img_cond_embeds.penultimate_hidden_states.shape}")
            print( f"last_hidden_state shape {img_cond_embeds.last_hidden_state.shape}")
            print( f"image_embeds shape {img_cond_embeds.image_embeds.shape}")

            if is_plus:
                img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            else:
                img_cond_embeds = img_cond_embeds.image_embeds
            print( f"Embed shape {img_cond_embeds.shape}")
            
            inbetween_embeds = []
            # Make sure we have 2 images
            if len(img_cond_embeds) > 1:
                num_embeds = len(img_cond_embeds)
                # Add beggining buffer
                inbetween_embeds.extend([img_cond_embeds[0]] * buffer)
                # Interpolate embeds
                for i in range(len(img_cond_embeds) - 1):
                    embed1 = img_cond_embeds[i]
                    embed2 = img_cond_embeds[i + 1]
                    alphas = torch.linspace(0, 1, transitioning_frames)
                    for alpha in alphas:
                        eased_alpha = easing_function(alpha.item())
                        print(f"eased alpha {eased_alpha}")
                        inbetween_embed = (1 - eased_alpha) * embed1 + eased_alpha * embed2
                        inbetween_embeds.extend([inbetween_embed] * repeat_count)
                # Add ending buffer
                inbetween_embeds.extend([img_cond_embeds[-1]] * buffer)
                # Find size of batch
                batch_size = len(inbetween_embeds)

            inbetween_embeds = torch.stack(inbetween_embeds, dim=0)
            output.append(inbetween_embeds)

        # Create empty neg embeds
        if is_plus:
            img_uncond_embeds = clip_vision.encode_image(torch.zeros([1, 224, 224, 3])).penultimate_hidden_states
        else:
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        
        # Work out which frames to drop
        frames_to_drop = []
        if num_embeds > 2:
            for i in range(num_embeds-2):
                frames_to_drop.append(transitioning_frames*(i+1)+buffer-1)
        print(f"Frames to drop {frames_to_drop}")

        # Combine and format prompt strings
        def format_text_prompts(text_prompts):
            string = ""
            index = buffer
            for prompt in text_prompts:
                string += f"\"{index}\":\"{prompt}\",\n"
                index += transitioning_frames
            return string
        
        positive_string = format_text_prompts(positive_prompts)
        negative_string = format_text_prompts(negative_prompts)
        
        return (output[0], output[1], output[2], img_uncond_embeds, positive_string, negative_string, ipadapter, batch_size, frames_to_drop,)

class IG_CrossFadeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "input_images": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "repeat_count": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        }
    } 
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = TREE_INTERP

    def main(self, input_images, transitioning_frames, interpolation, repeat_count):

        # Assuming input_images is a list of tensors with shape [C, H, W]
        # Initialize an empty list to hold crossfaded images
        crossfade_images = []
        image_count = len(input_images)

        for i in range(image_count - 1):  # For each pair of images
            image1 = input_images[i]
            image2 = input_images[i + 1]
            for repeat in range(repeat_count - transitioning_frames):  # Repeat the current image
                crossfade_images.append(image1)
            alphas = torch.linspace(1.0 / (transitioning_frames + 1.0), 1.0 - 1.0 / (transitioning_frames + 1.0), transitioning_frames + 1)
            for alpha in alphas:  # Transition to the next image
                easing_function = easing_functions[interpolation]
                eased_alpha = easing_function(alpha.item())
                crossfaded_image = crossfade(image1, image2, eased_alpha)
                crossfade_images.append(crossfaded_image)

        # Handle the last image repetition
        for repeat in range(repeat_count):
            crossfade_images.append(input_images[-1])
        # crossfade_images.append(last_image)

        crossfade_images = torch.stack(crossfade_images, dim=0)
        return (crossfade_images, )
