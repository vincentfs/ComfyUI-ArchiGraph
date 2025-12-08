CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/🛠️ Utils" # organized by py file

import os
import random

import torch
from comfy.utils import common_upscale
import numpy as np
from .anyType import ANY
from .nodes_opencv import Nparray2Image

# import comfyui core node
from nodes import SaveImage 
import folder_paths
NPARRAY_DIR = os.path.join(folder_paths.get_output_directory(), "nparrays") 
JSON_DIR = os.path.join(folder_paths.get_output_directory(), "jsons")

#---------------------------------------------------------------------------------------------------------------------#
class JsonSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (ANY, {
                    "tooltip": "Input dictionary/list/tuple to save as JSON."
                }),
                "filename_noExt": ("STRING", {
                    "default": "unknown",
                    "tooltip": "Filename without extension (extension .json will be added automatically).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    NAME = "JSON Save"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Saves a dictionary/list/tuple as a .json file. Nparrays and Tensors will be converted to lists."

    def convert(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def execute(self, input, filename_noExt):
        import json

        if not os.path.exists(JSON_DIR):
            os.makedirs(JSON_DIR)

        file_path = os.path.join(JSON_DIR, filename_noExt + ".json")
        with open(file_path, 'w') as json_file:
            json.dump(input, json_file, indent=4, default=self.convert)

        return (file_path,)
    
#---------------------------------------------------------------------------------------------------------------------#
class JsonLoad:
    @classmethod
    def INPUT_TYPES(cls):
        if not os.path.exists(JSON_DIR):
            os.makedirs(JSON_DIR)
            files = []
        else:
            files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith('.json')])

        if len(files) == 0:
            files = ["None"]  # Placeholder to avoid empty list issues
        
        return {
            "required": {
                "jsonfile": (files, ),
                "filename_noExt": ("STRING", {
                    "default": "",
                    "tooltip": "Filename without extension (extension .json will be added automatically). If specified, npyfile will be ignored.",
                }),
            },
        }
    
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    NAME = "JSON Load"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Loads a dictionary/list/tuple from a .json file."

    def execute(self, jsonfile, filename_noExt=""):
        import json

        if filename_noExt != "":
            jsonfile = filename_noExt + ".json"
        else:
            if jsonfile == "None":
                raise FileNotFoundError("JSON files not found in the jsons directory.")

        file_path = os.path.join(JSON_DIR, jsonfile)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        return (data,)

#---------------------------------------------------------------------------------------------------------------------#
class NparraySave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparray": ("NPARRAY", {
                    "lazy": False,
                }),
                "filename_noExt": ("STRING", {
                    "default": "unknown",
                    "tooltip": "Filename without extension (extension .npy will be added automatically).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    NAME = "Nparray Save"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Saves a numpy array as a .npy file."

    def execute(self, nparray, filename_noExt):
        if not os.path.exists(NPARRAY_DIR):
            os.makedirs(NPARRAY_DIR)

        file_path = os.path.join(NPARRAY_DIR, filename_noExt + ".npy")
        np.save(file_path, nparray)
        return (file_path,)

#---------------------------------------------------------------------------------------------------------------------#
class NparrayLoad:
    @classmethod
    def INPUT_TYPES(cls):
        if not os.path.exists(NPARRAY_DIR):
            os.makedirs(NPARRAY_DIR)
            files = []
        else:
            files = sorted([f for f in os.listdir(NPARRAY_DIR) if f.endswith('.npy')])

        if len(files) == 0:
            files = ["None"]  # Placeholder to avoid empty list issues
        
        return {
            "required": {
                "npyfile": (files, ),
                "filename_noExt": ("STRING", {
                    "default": "",
                    "tooltip": "Filename without extension (extension .npy will be added automatically). If specified, npyfile will be ignored.",
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("nparray",)
    FUNCTION = "execute"
    NAME = "Nparray Load"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Loads a numpy array from a .npy file."

    def execute(self, npyfile, filename_noExt=""):
        if filename_noExt != "":
            npyfile = filename_noExt + ".npy"
        else:
            if npyfile == "None":
                raise FileNotFoundError("Npy files not found in the nparrays directory.")

        file_path = os.path.join(NPARRAY_DIR, npyfile)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Npy file not found: {file_path}")

        nparray = np.load(file_path)

        return (nparray,)

#---------------------------------------------------------------------------------------------------------------------#
class SaveNparrayImage(SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {"tooltip": "The nparray images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "execute"
    NAME = "Save Nparray Image"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Saves the input nparray images to your ComfyUI output directory."

    def execute(self, nparrays, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        images = Nparray2Image().execute(nparrays)[0]
        return super().save_images(images, filename_prefix, prompt, extra_pnginfo)

#---------------------------------------------------------------------------------------------------------------------#
class PreviewNparrayImage(SaveNparrayImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"nparrays": ("NPARRAY", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    
    NAME = "Preview Nparray Image"
    DESCRIPTION = "Previews the input nparray images by saving them to a temporary directory."

#---------------------------------------------------------------------------------------------------------------------#
class Printf:
    '''
    Later can be used to unravel list and show specified elements only.
    '''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (ANY, {
                    "forceInput": True,
                    "tooltip": "Select objects to print."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    OUTPUT_NODE = True
    # INPUT_IS_LIST = True # if the input is a list, workflow will pass each element separately
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    NAME = "Printf"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Print object info to console for debugging. Supports any input type." 

    def execute(self, any, unique_id=None, extra_pnginfo=None):
        # prepare text, caution: text is parse at frontend as list of strings
        text = []
        if isinstance(any, np.ndarray):
            text.append(f"Type: Nparray, Shape: {any.shape}, Dtype: {any.dtype}")
        elif isinstance(any, torch.Tensor):
            text.append(f"Type: Tensor, Shape: {any.shape}, Dtype: {any.dtype}, Device: {any.device}")
        elif isinstance(any, list):
            text.append(f"Type: List, Length: {len(any)}")
            for i, item in enumerate(any):
                text.append(f"[{i}]: Type: {type(item)}, Value:\n{item}")
        elif isinstance(any, dict):
            text.append(f"Type: Dict, Length: {len(any)}")
            for key, value in any.items():
                text.append(f"Key: {key}, Type: {type(value)}, Value:\n{value}")
        elif isinstance(any, tuple):
            text.append(f"Type: Tuple, Length: {len(any)}")
            for i, item in enumerate(any):
                text.append(f"[{i}]: Type: {type(item)}, Value:\n{item}")
        else:
            text.append(f"Type: {type(any)}, Value:\n{any}")

        # save to workflow pnginfo
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

#---------------------------------------------------------------------------------------------------------------------#
class SliceList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": (ANY, {
                    "tooltip": "Input list to slice."
                }),
                "start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start index for slicing."
                }),
                "length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Length for slicing."
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("sliced",)
    FUNCTION = "execute"
    NAME = "Slicing List"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Extract a list from the input list from start index with specified length. Supports any input type." 

    def execute(self, input_list, start, length):
        if not (isinstance(input_list, list) or isinstance(input_list, tuple)):
            raise ValueError("Input is not a list or tuple.")

        sliced = input_list[start : start + length]

        return (sliced,)

#---------------------------------------------------------------------------------------------------------------------#
class SliceSublist:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": (ANY, {
                    "tooltip": "Input list to slice."
                }),
                "start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start index for slicing."
                }),
                "length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Length for slicing."
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("sliced",)
    FUNCTION = "execute"
    NAME = "Slicing Sublist"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Extract a sublist from the input list from start index with specified length. Supports any input type." 

    def execute(self, input_list, start, length):
        if not (isinstance(input_list, list) or isinstance(input_list, tuple)):
            raise ValueError("Input is not a list or tuple.")
        
        sliced = []
        
        for sublist in input_list:
            if not (isinstance(sublist, list) or isinstance(sublist, tuple)):
                sliced.append([])  # append empty list if sublist is not a list
            else:
                sliced.append(sublist[start : start + length])
        
        return (sliced,)
    
#---------------------------------------------------------------------------------------------------------------------#
class ImageResizeTo:
    upscale_methods = ["lanczos", "nearest-exact", "bilinear", "area", "bicubic"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",{
                    "lazy": False # Will only be evaluated if check_lazy_status requires it
                }),
                "edge_size": ("INT", {
                    "default": 1024, 
                    "min": 16, #Minimum value
                    "max": 8192, #Maximum value
                    "step": 2, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                }),
                "edge_type": ("BOOLEAN", {
                    "default": True, # Default value
                    "label_on": "By Short Edge", # Label for the first option
                    "label_off": "By Long Edge", # Label for the second option
                }),
                # "is_by_short": ("BOOL", {

                # }),
                "upscale_method": (cls.upscale_methods,),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "prev_width", "prev_height", "log")
    FUNCTION = "resize"
    NAME = "Image Resize To"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Resizes an image by specify one edge size. Proportion is kept by default."

    def resize(self, image, edge_size, edge_type, upscale_method):

        # get original image size
        B, H, W, C = image.shape
        prev_width = W
        prev_height = H

        image = image.movedim(-1, 1)

        if edge_type:
            if H < W:
                new_height = edge_size
                new_width = int(W * edge_size / H)
            else:
                new_width = edge_size
                new_height = int(H * edge_size / W)
        else:
            if H > W:
                new_height = edge_size
                new_width = int(W * edge_size / H)
            else:
                new_width = edge_size
                new_height = int(H * edge_size / W)

        image = common_upscale(image, new_width, new_height, upscale_method, "disabled")
        image = image.movedim(1, -1)

        log = f"Resized from W{prev_width}xH{prev_height} to W{new_width}xH{new_height}."

        return (image, prev_width, prev_height, log,)

#---------------------------------------------------------------------------------------------------------------------#
class Image2BW:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",{
                    "lazy": False
                }),
                "reds":("INT", {"default": 40, "min": -200, "max": 300, "display": "slider"}),
                "yellows":("INT", {"default": 60, "min": -200, "max": 300, "display": "slider"}),
                "greens":("INT", {"default": 40, "min": -200, "max": 300, "display": "slider"}),
                "cyans":("INT", {"default": 60, "min": -200, "max": 300, "display": "slider"}),
                "blues":("INT", {"default": 20, "min": -200, "max": 300, "display": "slider"}),
                "magentas":("INT", {"default": 80, "min": -200, "max": 300, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "bw"
    NAME = "Image to BW"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Converts an image to black & white like the adjustments in Photoshop."

    def bw(self, image, reds, yellows, greens, cyans, blues, magentas):

        datatype = torch.float32
        # get original image size
        B, H, W, C = image.shape

        color_ratios = torch.tensor([reds, greens, blues, cyans, magentas, yellows], dtype=datatype)
        color_ratios = (color_ratios + 200) / 500

        sorted_vals, sorted_idx = torch.sort(image, dim=3, descending=True) # BxHxWxC

        # Get the color ratio value of the maximum channel
        max_idx = sorted_idx[:, :, :, 0].unsqueeze(-1) # BxHxWx1
        ratios_reshaped = color_ratios.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,H,W,-1) # BxHxWx6
        ratio_max = torch.gather(ratios_reshaped, 3, max_idx) # BxHxWx1

        combo_idx = (sorted_idx[:, :, :, -1] + 3).unsqueeze(-1) # BxHxWx1
        ratio_combo = torch.gather(ratios_reshaped, 3, combo_idx) # BxHxWx1

        # duplicate sorted_vals to a new tensor
        sorted_vals_2 = sorted_vals.clone()
        sorted_vals_2[:, :, :, 0] = 0

        # move the first value to the last in sorted_vals_2
        sorted_vals_2 = torch.cat((sorted_vals_2[:, :, :, 1:], sorted_vals_2[:, :, :, 0].unsqueeze(-1)), dim=-1) # BxHxWxC
        
        diff = sorted_vals - sorted_vals_2 # BxHxWxC
        ratio = torch.cat((ratio_max, ratio_combo, torch.ones_like(ratio_max, dtype=datatype)), dim=-1) # BxHxWx3

        # calculate the sum product of diff and ratio
        final = torch.sum(diff * ratio,dim=-1) # BxHxWx1

        return (final, )
