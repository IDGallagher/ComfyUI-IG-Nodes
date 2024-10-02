import os
import hashlib
import numpy as np
import torch
import math
import json
import torch.nn.functional as F

import sys
from comfy import model_management
import folder_paths
from ..common.tree import *
from ..common.constants import *
from ..motion_predictor import MotionPredictor
import comfy.utils

# New imports for plotting
import matplotlib.pyplot as plt
import io

import os
import hashlib
import numpy as np
import torch
import math
import json
import torch.nn.functional as F

import sys
from comfy import model_management
import folder_paths
from ..common.tree import *
from ..common.constants import *
from ..motion_predictor import MotionPredictor
import comfy.utils

# New imports for plotting
import matplotlib.pyplot as plt
import io

class SM_VideoBase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "base_latents": ("LATENT",),
                "vae": ("VAE", ),
                "video_control": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "base_weight": ("FLOAT", {"default": 1.0}),
            }, 
            "optional": {
                "video_frames_1": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_2": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_3": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_4": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_5": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_6": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_7": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_8": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_9": ("IMAGE", {"default": None, "forceInput": True}),
                "video_frames_10": ("IMAGE", {"default": None, "forceInput": True}),
            }
        }

    # Updated RETURN_TYPES and RETURN_NAMES to include the weight plot as a tensor
    RETURN_TYPES = ("LATENT", "INT", "IMAGE",)
    RETURN_NAMES = ("latents", "batch_count", "weights_plot",)
    FUNCTION = "main"
    CATEGORY = TREE_SM

    @torch.inference_mode()
    def main(self, base_latents, vae, video_control, base_weight=1.0, 
             video_frames_1=None, video_frames_2=None, video_frames_3=None, 
             video_frames_4=None, video_frames_5=None, video_frames_6=None, 
             video_frames_7=None, video_frames_8=None, video_frames_9=None, 
             video_frames_10=None):
        """
        Combines base_latents with video-induced latents based on video_control parameters
        and generates a consolidated plot of the resultant weights as a tensor.
        
        Args:
            base_latents (dict): The base latent representations containing the tensor in `samples`.
            vae (VAE): The Variational Autoencoder model to encode video frames.
            video_control (str): Multiline string specifying video controls.
            base_weight (float): Weight for the base latents.
            video_frames_x (torch.Tensor or None): Optional video frames to be combined. Each should be a tensor of shape (batch, channels, height, width)
        
        Returns:
            tuple: (combined_latents, batch_count, weights_plot_tensor)
        """
        # --- Base Latents Processing ---
        if not isinstance(base_latents, dict) or 'samples' not in base_latents:
            raise TypeError("base_latents must be a dict with a 'samples' key containing the tensor.")
        
        base_latents_tensor = base_latents['samples']  # Access the tensor
        print(f"[DEBUG] base_latents_tensor shape: {base_latents_tensor.shape}")

        # Extract target height and width from base_latents_tensor
        H_b, W_b = base_latents_tensor.shape[2], base_latents_tensor.shape[3]
        print(f"[DEBUG] Target spatial dimensions - Height: {H_b}, Width: {W_b}")

        # Initialize list to hold latents from video frames
        video_latents = [None] * 10  # Initialize to None for up to 10 videos

        # List of video_frames for easy iteration
        video_frames = [
            video_frames_1, video_frames_2, video_frames_3, video_frames_4, video_frames_5,
            video_frames_6, video_frames_7, video_frames_8, video_frames_9, video_frames_10
        ]

        # Encode each provided video frame to latents
        for idx, frame in enumerate(video_frames):
            if frame is not None:
                # Ensure frame is a 4D tensor: (batch, channels, height, width)
                if isinstance(frame, torch.Tensor):
                    if frame.dim() == 3:
                        # Single image: (channels, height, width) -> add batch dimension
                        frame = frame.unsqueeze(0)
                    elif frame.dim() == 4:
                        # Already batched
                        pass
                    else:
                        raise ValueError(f"video_frames_{idx+1} has unsupported number of dimensions: {frame.dim()}")
                else:
                    raise TypeError(f"video_frames_{idx+1} should be a torch.Tensor, got {type(frame)}")

                # Encode the frame to latent space
                video_latents[idx] = vae.encode(frame)  # Shape: (batch_size, latent_dim, H_v, W_v)
                print(f"[DEBUG] video_latents[{idx}] shape after encoding: {video_latents[idx].shape}")

                # Resize the video latent to match base_latents_tensor's spatial dimensions if necessary
                H_v, W_v = video_latents[idx].shape[2], video_latents[idx].shape[3]
                if (H_v != H_b) or (W_v != W_b):
                    video_latents[idx] = F.interpolate(video_latents[idx], size=(H_b, W_b), mode='bilinear', align_corners=False)
                    print(f"[DEBUG] video_latents[{idx}] resized to: {video_latents[idx].shape}")

        # --- Video Control Parsing ---
        video_controls = []
        lines = video_control.strip().split('\n')
        for idx, line in enumerate(lines[:10]):  # Limit to 10 controls
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    # Parse start_frame and end_frame
                    start_frame = int(parts[0].strip())
                    end_frame = int(parts[1].strip())

                    if end_frame <= start_frame:
                        raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame}).")

                    # Parse mid_start_fraction and mid_end_fraction
                    mid_fractions_str = ','.join(parts[2:4]).strip()
                    if mid_fractions_str.startswith('(') and mid_fractions_str.endswith(')'):
                        mid_fractions = mid_fractions_str[1:-1].split(',')
                        if len(mid_fractions) != 2:
                            raise ValueError("Mid fractions must contain exactly two float values.")
                        mid_start_frac = float(mid_fractions[0].strip())
                        mid_end_frac = float(mid_fractions[1].strip())
                    else:
                        raise ValueError("Mid fractions must be enclosed in parentheses and separated by a comma.")

                    # Validate fractions
                    if not (0.0 <= mid_start_frac < mid_end_frac <= 1.0):
                        raise ValueError("Mid fractions must satisfy 0.0 <= mid_start_frac < mid_end_frac <= 1.0.")

                    # Parse weights
                    weights_str = ','.join(parts[4:7]).strip()
                    if weights_str.startswith('(') and weights_str.endswith(')'):
                        weights = weights_str[1:-1].split(',')
                        if len(weights) != 3:
                            raise ValueError("Weights must contain exactly three float values.")
                        start_weight = float(weights[0].strip())
                        mid_weight = float(weights[1].strip())
                        end_weight = float(weights[2].strip())
                    else:
                        raise ValueError("Weights must be enclosed in parentheses and separated by a comma.")

                    # Optionally, validate weight ranges if necessary
                    # For example, weights between 0.0 and 1.0
                    for weight, name in zip([start_weight, mid_weight, end_weight], 
                                            ['start_weight', 'mid_weight', 'end_weight']):
                        if not (0.0 <= weight <= 1.0):
                            raise ValueError(f"{name} ({weight}) must be between 0.0 and 1.0.")

                    video_controls.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'mid_start_frac': mid_start_frac,
                        'mid_end_frac': mid_end_frac,
                        'start_weight': start_weight,
                        'mid_weight': mid_weight,
                        'end_weight': end_weight,
                        'video_index': idx  # Mapping to video_latents index
                    })

                except ValueError as e:
                    # Raise an error with a clear message
                    raise ValueError(
                        f"Invalid format in video_control line {idx+1}: '{line}'.\n"
                        f"Expected format: start_frame, end_frame, (mid_start_fraction, mid_end_fraction), "
                        f"(start_weight, mid_weight, end_weight)\n"
                        f"Error Details: {e}"
                    )
            else:
                # If not enough parameters, raise an error with a clear message
                raise ValueError(
                    f"Invalid format in video_control line {idx+1}: '{line}'.\n"
                    f"Expected format: start_frame, end_frame, (mid_start_fraction, mid_end_fraction), "
                    f"(start_weight, mid_weight, end_weight)"
                )
        print(f"[DEBUG] Parsed video_controls: {video_controls}")

        # --- Combination of Latents ---
        total_frames = base_latents_tensor.shape[0]
        print(f"[DEBUG] total_frames: {total_frames}")

        # Initialize the combined latents with base_latents multiplied by base_weight
        combined_latents = base_latents_tensor * base_weight  # Updated: Use base_latents_tensor
        print(f"[DEBUG] combined_latents initialized with shape: {combined_latents.shape}")

        # Initialize a tensor to keep track of total weights per frame
        total_weights = torch.full((total_frames, 1, 1, 1), base_weight, device=base_latents_tensor.device)  # Updated: Use base_latents_tensor.device
        print(f"[DEBUG] total_weights initialized with shape: {total_weights.shape}")

        # Collect weights for plotting
        plot_data = []  # List of dictionaries: {'video_line': X, 'weights': [...], 'start_frame': Y, 'end_frame': Z}

        # Iterate over each video control to accumulate weighted latents
        for control in video_controls:
            start = control['start_frame']
            end = control['end_frame']
            mid_start_frac = control['mid_start_frac']
            mid_end_frac = control['mid_end_frac']
            start_weight = control['start_weight']
            mid_weight = control['mid_weight']
            end_weight = control['end_weight']
            vid_idx = control['video_index']

            print(f"[DEBUG] Processing video_control line {vid_idx+1}: start={start}, end={end}, "
                  f"mid_start_frac={mid_start_frac}, mid_end_frac={mid_end_frac}, "
                  f"start_weight={start_weight}, mid_weight={mid_weight}, end_weight={end_weight}, "
                  f"vid_idx={vid_idx}")

            # Check if the corresponding video_latents are available
            if vid_idx >= len(video_latents) or video_latents[vid_idx] is None:
                raise ValueError(
                    f"Video_control line {vid_idx+1} refers to video_frames_{vid_idx+1}, "
                    f"which was not provided or could not be encoded."
                )

            vid_latents = video_latents[vid_idx]  # Shape: (batch_size, latent_dim, H_b, W_b) after resizing
            print(f"[DEBUG] video_latents[{vid_idx}] shape: {vid_latents.shape}")

            # Calculate the frame range overlap
            applicable_start = max(start, 0)
            applicable_end = min(end, total_frames)
            num_applicable_frames = applicable_end - applicable_start

            print(f"[DEBUG] Applicable frame range: {applicable_start} to {applicable_end} (num_applicable_frames={num_applicable_frames})")

            if num_applicable_frames <= 0:
                raise ValueError(
                    f"Video_control line {vid_idx+1} has no overlapping frames with base_latents."
                )

            # Compute actual mid_start_frame and mid_end_frame based on fractions
            slice_length = applicable_end - applicable_start
            mid_start_frame = applicable_start + int(math.floor(mid_start_frac * slice_length))
            mid_end_frame = applicable_start + int(math.floor(mid_end_frac * slice_length))

            # Ensure mid_start_frame and mid_end_frame are within bounds
            mid_start_frame = max(applicable_start, min(mid_start_frame, applicable_end))
            mid_end_frame = max(mid_start_frame, min(mid_end_frame, applicable_end))

            print(f"[DEBUG] Computed mid_start_frame: {mid_start_frame}, mid_end_frame: {mid_end_frame}")

            # Define frame indices for the three phases
            # Create a weight tensor for the applicable frames
            weights = torch.zeros((num_applicable_frames,), device=base_latents_tensor.device, dtype=torch.float32)

            # Phase 1: start to mid_start_frame (transition from start_weight to mid_weight)
            if mid_start_frame > applicable_start:
                phase1_len = mid_start_frame - applicable_start
                if phase1_len > 1:
                    phase1 = torch.linspace(start_weight, mid_weight, steps=phase1_len, device=base_latents_tensor.device, dtype=torch.float32)
                else:
                    phase1 = torch.tensor([mid_weight], device=base_latents_tensor.device, dtype=torch.float32)
                weights[:phase1_len] = phase1
                print(f"[DEBUG] Phase 1: {phase1_len} frames, start_weight to mid_weight")
            else:
                print(f"[DEBUG] Phase 1 skipped for video_control line {vid_idx+1}")

            # Phase 2: mid_start_frame to mid_end_frame (constant at mid_weight)
            if mid_end_frame > mid_start_frame:
                phase2_len = mid_end_frame - mid_start_frame
                phase2_start = mid_start_frame - applicable_start
                phase2_end = phase2_start + phase2_len
                weights[phase2_start:phase2_end] = mid_weight
                print(f"[DEBUG] Phase 2: {phase2_len} frames at mid_weight")
            else:
                print(f"[DEBUG] Phase 2 skipped for video_control line {vid_idx+1}")

            # Phase 3: mid_end_frame to end (transition from mid_weight to end_weight)
            if applicable_end > mid_end_frame:
                phase3_len = applicable_end - mid_end_frame
                phase3_start = mid_end_frame - applicable_start
                phase3_end = phase3_start + phase3_len
                if phase3_len > 1:
                    phase3 = torch.linspace(mid_weight, end_weight, steps=phase3_len, device=base_latents_tensor.device, dtype=torch.float32)
                else:
                    phase3 = torch.tensor([end_weight], device=base_latents_tensor.device, dtype=torch.float32)
                weights[phase3_start:phase3_end] = phase3
                print(f"[DEBUG] Phase 3: {phase3_len} frames, mid_weight to end_weight")
            else:
                print(f"[DEBUG] Phase 3 skipped for video_control line {vid_idx+1}")

            # Reshape weights for broadcasting
            weights = weights.view(-1, 1, 1, 1)  # Shape: (num_applicable_frames, 1, 1, 1)
            print(f"[DEBUG] Computed weights shape: {weights.shape}")

            # **Print the resultant weights for this video**
            # Convert weights to CPU and list for readable printing
            weights_list = weights.squeeze().cpu().tolist()
            print(f"[INFO] Resultant weights for video_control line {vid_idx+1}: {weights_list}")

            # Collect weights data for plotting
            plot_data.append({
                'video_line': vid_idx + 1,
                'weights': weights_list,
                'start_frame': applicable_start,
                'end_frame': applicable_end
            })

            # Slice the video latents to match the number of applicable frames
            vid_latents_slice = vid_latents[:num_applicable_frames]
            print(f"[DEBUG] vid_latents_slice shape: {vid_latents_slice.shape}")
            print(f"[DEBUG] combined_latents slice shape: {combined_latents[applicable_start:applicable_end].shape}")

            # Check shape compatibility
            if vid_latents_slice.shape != combined_latents[applicable_start:applicable_end].shape:
                raise ValueError(
                    f"Shape mismatch between video_latents_slice and base_latents slice for video_control line {vid_idx+1}.\n"
                    f"video_latents_slice shape: {vid_latents_slice.shape}\n"
                    f"base_latents slice shape: {combined_latents[applicable_start:applicable_end].shape}"
                )

            # Add the weighted video latents to the combined_latents
            combined_latents[applicable_start:applicable_end] += vid_latents_slice * weights
            print(f"[DEBUG] Updated combined_latents[{applicable_start}:{applicable_end}] with weighted video_latents_slice.")

            # Update the total_weights
            total_weights[applicable_start:applicable_end] += weights
            print(f"[DEBUG] Updated total_weights[{applicable_start}:{applicable_end}] with weights.")

        # --- Normalization ---
        print(f"[DEBUG] combined_latents shape before normalization: {combined_latents.shape}")
        print(f"[DEBUG] total_weights shape: {total_weights.shape}")

        with torch.no_grad():
            normalized_latents = combined_latents / total_weights
            # Create a mask where total_weights == 0
            zero_weight_mask = (total_weights == 0).expand_as(combined_latents)
            # Replace normalized_latents with base_latents_tensor where zero_weight_mask is True
            normalized_latents = torch.where(zero_weight_mask, base_latents_tensor, normalized_latents)

        print(f"[DEBUG] normalized_latents shape: {normalized_latents.shape}")
        print(f"[DEBUG] zero_weight_mask count: {(zero_weight_mask).sum().item()}")

        # --- Generate the Consolidated Weights Plot as a Tensor ---
        if plot_data:
            plt.figure(figsize=(12, 8))
            
            # Define a color palette with distinct colors
            color_palette = plt.get_cmap('tab10').colors  # 10 distinct colors
            num_colors = len(color_palette)

            for idx, data in enumerate(plot_data):
                frames = list(range(data['start_frame'], data['end_frame']))
                weight_values = data['weights']
                
                # Assign a color from the palette, cycling if necessary
                color = color_palette[idx % num_colors]
                
                plt.plot(frames, weight_values, label=f'Video {data["video_line"]}', color=color)

            # Plot the base weight as a horizontal line
            plt.axhline(y=base_weight, color='black', linestyle='--', label='Base Weight')

            plt.title('Weights Distribution Across Frames')
            plt.xlabel('Frame')
            plt.ylabel('Weight')
            plt.xlim(0, total_frames)
            plt.ylim(0, 1.1)  # Assuming weights are between 0 and 1

            # Remove grid lines
            plt.grid(False)

            plt.legend()

            plt.tight_layout()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close()

            # Read the image from buffer as a numpy array
            image_np = plt.imread(buf)
            buf.close()

            # Convert the numpy array to a torch tensor
            # matplotlib.pyplot.imread returns a float32 array in [0,1] or uint8
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)

            # Handle cases where image has alpha channel or is grayscale
            if image_np.ndim == 2:
                # Grayscale image, replicate channels to make RGB
                image_np = np.stack([image_np]*3, axis=-1)
                print("[DEBUG] Converted grayscale image to RGB by replicating channels.")
            elif image_np.shape[2] == 4:
                # If RGBA, convert to RGB by dropping alpha channel
                image_np = image_np[:, :, :3]
                print("[DEBUG] Dropped alpha channel from RGBA image to convert to RGB.")
            elif image_np.shape[2] == 1:
                # If single channel, replicate channels to make RGB
                image_np = np.repeat(image_np, 3, axis=2)
                print("[DEBUG] Replicated single channel to make RGB.")

            image_tensor = torch.from_numpy(image_np).float() / 255.0  # Normalize to [0,1]

            print("[INFO] Consolidated weights plot tensor generated successfully.")
        else:
            # If no plot data, create a blank RGB image tensor
            image_tensor = torch.ones((600, 800, 3), dtype=torch.float32)  # RGB, H=600, W=800
            print("[INFO] No video controls provided. Generated an empty weights plot tensor.")
# IMAGE torch.Size([3, 770, 1170])
        image_tensor = image_tensor.unsqueeze(0)
        print(f"IMAGE {image_tensor.shape}")
        # --- Wrap the Normalized Latents ---
        combined_latents_dict = {'samples': normalized_latents}

        return combined_latents_dict, total_frames, image_tensor  # Return the dict, batch_count, and weights_plot_tensor
    

class SM_VideoBaseControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "base_steps": ("INT", {"default": 0}),
            }, 
            "optional": {
                "video_control": ("STRING", {"default": """# Video Control Format:
# Each line defines a control with the following comma-separated values:
# start_frame, end_frame, (mid_start_fraction, mid_end_fraction), (start_weight, mid_weight, end_weight)

# Example:
20, 40, (0.4,0.6), (0.5,1.0,0.5)
50, 150, (0.3,0.7), (0.3,0.8,0.3)
""", "multiline": True}),
            }
        }

    # Define RETURN_TYPES with video_length_1 to video_length_10
    RETURN_TYPES = ("STRING", "INT") + tuple(["INT"] * 10)
    # Define RETURN_NAMES accordingly
    RETURN_NAMES = (
        "video_control",
        "base_steps",
        "video_length_1",
        "video_length_2",
        "video_length_3",
        "video_length_4",
        "video_length_5",
        "video_length_6",
        "video_length_7",
        "video_length_8",
        "video_length_9",
        "video_length_10",
    )
    FUNCTION = "main"
    CATEGORY = TREE_SM

    @torch.inference_mode()
    def main(self, base_steps=0, video_control=""):
        video_lengths = [0] * 10  # Initialize all video lengths to 0
        lines = video_control.strip().split('\n')
        
        for idx, line in enumerate(lines[:10]):  # Limit to 10 lines
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    # Parse start_frame and end_frame
                    start_frame = int(parts[0].strip())
                    end_frame = int(parts[1].strip())

                    if end_frame <= start_frame:
                        raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame}).")

                    # Parse mid_start_fraction and mid_end_fraction
                    mid_fractions_str = ','.join(parts[2:4]).strip()
                    if mid_fractions_str.startswith('(') and mid_fractions_str.endswith(')'):
                        mid_fractions = mid_fractions_str[1:-1].split(',')
                        if len(mid_fractions) != 2:
                            raise ValueError("Mid fractions must contain exactly two float values.")
                        mid_start_frac = float(mid_fractions[0].strip())
                        mid_end_frac = float(mid_fractions[1].strip())
                    else:
                        raise ValueError("Mid fractions must be enclosed in parentheses and separated by a comma.")

                    # Validate fractions
                    if not (0.0 <= mid_start_frac < mid_end_frac <= 1.0):
                        raise ValueError("Mid fractions must satisfy 0.0 <= mid_start_frac < mid_end_frac <= 1.0.")

                    # Parse weights
                    weights_str = ','.join(parts[4:7]).strip()
                    if weights_str.startswith('(') and weights_str.endswith(')'):
                        weights = weights_str[1:-1].split(',')
                        if len(weights) != 3:
                            raise ValueError("Weights must contain exactly three float values.")
                        start_weight = float(weights[0].strip())
                        mid_weight = float(weights[1].strip())
                        end_weight = float(weights[2].strip())
                    else:
                        raise ValueError("Weights must be enclosed in parentheses and separated by a comma.")

                    # Optionally, validate weight ranges if necessary
                    # For example, weights between 0.0 and 1.0
                    for weight, name in zip([start_weight, mid_weight, end_weight], 
                                            ['start_weight', 'mid_weight', 'end_weight']):
                        if not (0.0 <= weight <= 1.0):
                            raise ValueError(f"{name} ({weight}) must be between 0.0 and 1.0.")

                    video_length = end_frame - start_frame
                    video_lengths[idx] = video_length

                except ValueError as e:
                    # Raise an error with a clear message
                    raise ValueError(
                        f"Invalid format in video_control line {idx+1}: '{line}'.\n"
                        f"Expected format: start_frame, end_frame, (mid_start_fraction, mid_end_fraction), "
                        f"(start_weight, mid_weight, end_weight)\n"
                        f"Error Details: {e}"
                    )
            else:
                # If not enough parameters, raise an error with a clear message
                raise ValueError(
                    f"Invalid format in video_control line {idx+1}: '{line}'.\n"
                    f"Expected format: start_frame, end_frame, (mid_start_fraction, mid_end_fraction), "
                    f"(start_weight, mid_weight, end_weight)"
                )

        # If fewer than 10 lines are provided, the remaining video_lengths stay at 0
        # Return as per RETURN_TYPES
        return (video_control, base_steps, *video_lengths)