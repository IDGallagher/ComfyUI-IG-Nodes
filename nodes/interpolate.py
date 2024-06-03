import os
import hashlib
import numpy as np
import torch
import math
import json

import sys
import folder_paths
from ..common.tree import *
from ..common.constants import *

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
    
        # If not at end, transition image
            

        # for i in range(transitioning_frames):
        #     alpha = alphas[i]
        #     image1 = images_1[i + transition_start_index]
        #     image2 = images_2[i + transition_start_index]
        #     easing_function = easing_functions.get(interpolation)
        #     alpha = easing_function(alpha)  # Apply the easing function to the alpha value

        #     crossfade_image = crossfade(image1, image2, alpha)
        #     crossfade_images.append(crossfade_image)
            
        # # Convert crossfade_images to tensor
        # crossfade_images = torch.stack(crossfade_images, dim=0)
        # # Get the last frame result of the interpolation
        # last_frame = crossfade_images[-1]
        # # Calculate the number of remaining frames from images_2
        # remaining_frames = len(images_2) - (transition_start_index + transitioning_frames)
        # # Crossfade the remaining frames with the last used alpha value
        # for i in range(remaining_frames):
        #     alpha = alphas[-1]
        #     image1 = images_1[i + transition_start_index + transitioning_frames]
        #     image2 = images_2[i + transition_start_index + transitioning_frames]
        #     easing_function = easing_functions.get(interpolation)
        #     alpha = easing_function(alpha)  # Apply the easing function to the alpha value

        #     crossfade_image = crossfade(image1, image2, alpha)
        #     crossfade_images = torch.cat([crossfade_images, crossfade_image.unsqueeze(0)], dim=0)
        # # Append the beginning of images_1
        # beginning_images_1 = images_1[:transition_start_index]
        # crossfade_images = torch.cat([beginning_images_1, crossfade_images], dim=0)
        return (crossfade_images, )
    

# class IG_ParseqToWeights:

#     FUNCTION = "main"
#     CATEGORY = TREE_INTERP
#     RETURN_TYPES = ("FLOAT",)
#     RETURN_NAMES = ("weights",)

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "parseq": ("STRING", {"default": '', "multiline": True}),
#             },
#         } 

#     def main(self, parseq):
#         # Load the JSON string into a dictionary
#         data = json.loads(parseq)

#         # Extract the list of frames
#         frames = data.get('rendered_frames', [])

#         # Extract the prompt_weight_1 from each frame and store it in a list
#         prompt_weights = [frame['prompt_weight_1'] for frame in frames]

#         return (prompt_weights, )
