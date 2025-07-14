import sys
import os
import json
import random
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import cv2

from ..common.tree import *
from ..common.constants import *
from .io import is_changed_load_images

class IG_ExplorerNode:
    
    def __init__(self) -> None:
        self.folder = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"forceInput": True}),
                "precision": ("FLOAT", {"default": 0.0001, "min": FLOAT_STEP, "max": 1, "step": FLOAT_STEP}),
                "max_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "num_explorations": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_seed": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, folder: str, precision, max_images, num_explorations, start_seed, **kwargs):
        print(f"IS CHANGED")
        # return random.randint(1, 100000000000)
        return float("NaN")

    RETURN_TYPES = ("FLOAT","STRING","INT",)
    FUNCTION = "main"
    CATEGORY = TREE_EXPLORER

    class Job:
        def __init__(self, param1, param2, difference=None):
            self.param1 = param1
            self.param2 = param2
            self.difference = difference

    def image_filename(self, param):
        # Format the parameter to have six digits before the decimal and six digits after
        param_str = "{:02d}_{:024d}".format(int(param), int((param % 1) * 1000000000000000000000000))
        return os.path.join(self.folder, f"image_{param_str}")
    
    # def image_filename(self, param):
        # Format the parameter to a fixed length with zero-padding
        # param_str = "{:06d}".format(int(param / FLOAT_STEP))  # Assuming param is a float
        # return os.path.join(self.folder, f"image_{param_str}")
    
    def image_exists(self, param):
        full_file = self.image_filename(param) + "_00001_.png"
        print(f"Full file: {full_file}")
        return os.path.isfile(full_file)
    
    def count_images(self):
        return len([f for f in os.listdir(self.folder) if f.startswith("image_") and f.endswith("_00001_.png") and os.path.isfile(os.path.join(self.folder, f))])

    def load_state(self, path, start_seed):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {'current_seed': start_seed, 'explorations_done': 0, 'active_subfolder': None}

    def save_state(self, path, state):
        with open(path, 'w') as f:
            json.dump(state, f)

    def save_queue(self, queue, filename):
        # Use the .txt extension or any other preferred plaintext format
        filename = os.path.splitext(filename)[0] + '.txt'

        with open(filename, 'w') as f:
            for job in queue:
                # Create a list of job attributes
                job_data = [str(job.param1), str(job.param2)]
                if job.difference is not None:
                    job_data.append(str(job.difference))
                
                # Join the attributes with a tab delimiter and write to the file
                f.write('\t'.join(job_data) + '\n')

    def load_queue(self, filename):
        # Change the file extension from .json to .txt to match the new format
        filename = os.path.splitext(filename)[0] + '.txt'
        
        try:
            with open(filename, 'r') as f:
                queue = []
                for line in f:
                    parts = line.strip().split('\t')
                    param1, param2 = float(parts[0]), float(parts[1])
                    difference = float(parts[2]) if len(parts) > 2 else None
                    queue.append(self.Job(param1, param2, difference))
                return queue
        except FileNotFoundError:
            # Initialize with one job with parameters 0 and 1
            return [self.Job(0, 1)]

    # def measure_difference(self, param1, param2):
        # return 1
    
    def measure_difference(self, param1, param2):
        # Read the images
        image1 = imread(self.image_filename(param1) + "_00001_.png")
        image2 = imread(self.image_filename(param2) + "_00001_.png")
        
        # Convert the images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between two images
        ssim_value = ssim(image1_gray, image2_gray)
        print(f"{param1} {param2} - {ssim_value}")
        return ssim_value

    def process_jobs(self, queue, precision, max_images, current_count):
        for job in queue:
            if job.difference is None:
                if not self.image_exists(job.param1):
                    if max_images == 0 or current_count < max_images:
                        return job.param1
                    else:
                        continue
                if not self.image_exists(job.param2):
                    if max_images == 0 or current_count < max_images:
                        return job.param2
                    else:
                        continue
                job.difference = self.measure_difference(job.param1, job.param2)

        eligible_jobs = [job for job in queue if job.difference is not None and abs(job.param1 - job.param2) > precision]

        if not eligible_jobs:
            return -1

        if max_images != 0 and current_count >= max_images:
            return -1

        # Find the job with the maximum difference
        max_diff_job = min(eligible_jobs, key=lambda job: job.difference)
        print(f"Min similarity {max_diff_job.param1} {max_diff_job.param2} {max_diff_job.difference}")
        queue.remove(max_diff_job)

        # Split range and create new jobs
        mid_param = (max_diff_job.param1 + max_diff_job.param2) / 2
        new_job1 = self.Job(max_diff_job.param1, mid_param)
        new_job2 = self.Job(mid_param, max_diff_job.param2)
        queue.extend([new_job1, new_job2])

        # Generate the new image
        return mid_param

    def main(self, folder, precision, max_images, num_explorations, start_seed):
        os.makedirs(folder, exist_ok=True)
        state_path = os.path.join(folder, 'exploration_state.json')
        while True:
            state = self.load_state(state_path, start_seed)
            if state['active_subfolder'] is None:
                if num_explorations > 0 and state['explorations_done'] >= num_explorations:
                    raise AssertionError("🤖 All explorations finished!")
                current_seed = state['current_seed']
                subfolder = os.path.join(folder, f"seed_{current_seed}")
                os.makedirs(subfolder, exist_ok=True)
                state['active_subfolder'] = subfolder
                state['current_seed'] = current_seed + 1
                state['explorations_done'] += 1
                self.save_state(state_path, state)
            self.folder = state['active_subfolder']
            queue_path = os.path.join(self.folder, 'job_queue.json')
            queue = self.load_queue(queue_path)
            current_count = self.count_images()
            param = self.process_jobs(queue, precision, max_images, current_count)
            self.save_queue(queue, queue_path)
            if param > -1:
                seed = int(os.path.basename(self.folder).split('_')[1])
                return (param, self.image_filename(param), seed)
            else:
                state['active_subfolder'] = None
                self.save_state(state_path, state)
