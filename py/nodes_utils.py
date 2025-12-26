CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/🛠️ Utils" # organized by py file

import os
import random, uuid, hashlib, time

import torch
import numpy as np
from PIL import Image
from .anyType import ANY
from .nodes_opencv import Nparray2Image
from ..node_registry import comfy_node

# import comfyui core node
from server import PromptServer # import comfyui server module to manupulate prompt queue
from nodes import SaveImage 
import folder_paths
NPARRAY_DIR = os.path.join(folder_paths.get_output_directory(), "nparrays") 
JSON_DIR = os.path.join(folder_paths.get_output_directory(), "jsons")

#---------------------------------------------------------------------------------------------------------------------#
# Helpers
#---------------------------------------------------------------------------------------------------------------------#
def resolve_path(path: str) -> str:      
    # 1. 展开环境变量（如 %USERPROFILE%）
    path = os.path.expandvars(path)
    # 2. 展开用户目录（如 ~ 表示用户主目录）
    path = os.path.expanduser(path)
    # 3. 转换为绝对路径并规范化
    path = os.path.abspath(path)
    return path

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
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
@comfy_node
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
@comfy_node
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
@comfy_node
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
@comfy_node
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
@comfy_node
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
@comfy_node
class Printf:
    '''
    TODO: Can be used to unravel list and show specified elements only.
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
    INPUT_IS_LIST = False # if the input is a list, workflow will pass each element separately
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

        return {"ui": {"text": text}, "result": (text,)}

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
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
@comfy_node
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
@comfy_node
class SaveImageToPath:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "target_folder": ("STRING", {
                    "default": "%USERPROFILE%",
                    "tooltip": "Target size for resizing."
                }),
                "filename": ("STRING", {
                    "default": "output",
                    "tooltip": "Filename without extension (extension .png will be added automatically).",
                }),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "execute"
    NAME = "Save Image To Path"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "TODO." 

    def execute(self, images, target_folder, filename):

        target_folder = resolve_path(target_folder)

        if not os.path.exists(target_folder):
            try:
                os.makedirs(target_folder, exist_ok=True)
            except Exception as e:
                print(f"AG {self.NAME}: Failed to create target folder: {target_folder}")
                return ()
            
        file_path = os.path.join(target_folder, f"{filename}.png")

        i = 255. * images[0].cpu().numpy()
        image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image.save(file_path, compress_level=4)
        print(f"AG {self.NAME}: {file_path} created.")
        return () # need to return something to avoid error in comfyui

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
class FilenameList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_files": ("INT", {
                    "default": 10,
                    "min": 1,
                    "tooltip": "Number of filenames to generate."
                }),
                "filename_prefix": ("STRING", {
                    "default": "AG",
                    "tooltip": "Prefix for the filenames."
                }),
                "num_digit": ("INT", {
                    "default": 4,
                    "tooltip": "Number of digits in the filename."
                }),
                "num_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting number for the filename."
                }),
            },
            "optional": {
                "optional_idx": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "If specified (not -1), generate only the filename for this index."
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("filenames",)
    FUNCTION = "execute"
    NAME = "Filename List"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Generates a list of filenames based on the given prefix and number format." 

    def execute(self, num_files: int, filename_prefix: str, num_digit: int, num_start: int, optional_idx: int = -1):
        if optional_idx == -1:
            filenames = [f"{filename_prefix}_{i:0{num_digit}}.png" for i in range(num_start, num_start + num_files)]
        else:
            filenames = f"{filename_prefix}_{optional_idx:0{num_digit}}.png"
        return (filenames,)

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
class FilepathGeneratorMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_folder": ("STRING", {
                    "default": "%USERPROFILE%",
                    "tooltip": "Target folder to enumerate files from. Subfolders are not included."
                }),
                "file_extension": ("STRING", {
                    "default": "",
                    "tooltip": "File extension to filter by. If empty, all 'png/jpg/jpeg' are included."
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("enumerator",)
    FUNCTION = "execute"
    NAME = "Filepath Generator Maker"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Returns a generator for filepaths from the specified target folder with the given file extension." 

    def execute(self, target_folder, file_extension):

        target_folder = resolve_path(target_folder)
        if not os.path.exists(target_folder):
            print(f"AG {self.NAME}: Target folder does not exist: {target_folder}")
            return (None,)

        from pathlib import Path

        def maker():
            nonlocal target_folder
            nonlocal file_extension
            def file_generator(folder, extension):
                path = Path(folder)
                if extension != "":
                    yield from path.rglob(f"*.{extension.lstrip('.')}")  # ensure extension starts with '.'
                else:
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        yield from path.rglob(ext)

            return file_generator(target_folder, file_extension)

        return (maker,)

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
class IterateItems:
    def __init__(self):
        self.index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": (ANY, {
                    "tooltip": "Input a list to iterate over its items."
                }),
                "iteration_on": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Always re-run this node, even if the input list is unchanged."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (ANY, "INT", )
    RETURN_NAMES = ("item", "index", )
    FUNCTION = "execute"
    NAME = "Iterate Items"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Iterate over each item in the input list. Outputs one item at a time." 

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False, )

    def execute(self, items, iteration_on, unique_id=None):
        # sanity check
        if not (isinstance(items, list) or isinstance(items, tuple)):
            print("Input is not a list or tuple, return as is.")
            return (items, 0, )
        
        # get list item
        item = items[self.index]
        print(f"AG {self.NAME}: iterated {self.index + 1} times.")

        # import json
        # print(json.dumps(current_item)) # for debugging

        # -----------------------------------------------------
        # hack the prompt queue to add a new item to the queue
        # -----------------------------------------------------
        if iteration_on:
            queue = PromptServer.instance.prompt_queue
            current_item = queue.currently_running[queue.task_counter - 1]

            # item structure: (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
            new_item = list(current_item) # copy current item
            new_item[0] = queue.task_counter # update index
            new_item[1] = str(uuid.uuid4()) # generate new prompt_id
            new_item[3]["create_time"] = int(time.time() * 1000)  # timestamp in milliseconds

            # add new item to queue
            if self.index < len(items) - 1:
                queue.put(new_item)
            
            current_idx = self.index
            self.index = (self.index + 1) % len(items) # will return to 0 when finished

            return (item, current_idx, )

        return (item, 0, )
    
    @classmethod
    def IS_CHANGED(cls, items, iteration_on, unique_id=None):
        if iteration_on:
            return float("nan") # for faster update

        input_str = f"{items}-{iteration_on}"
        return hashlib.sha256(input_str.encode("utf-8")).hexdigest()

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
class IterateGenerator:
    def __init__(self):
        self.index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generatorMaker": (ANY, {
                    "tooltip": "Input a generator to iterate"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only run when enabled."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (ANY, "INT")
    RETURN_NAMES = ("item", "index")
    FUNCTION = "execute"
    NAME = "Iterate Generator"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Iterate over each item from the generator. Outputs one item at a time." 

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False)

    def execute(self, generatorMaker, enabled, unique_id=None):
        # when not enabled, exit
        if not enabled:
            print(f"AG {self.NAME}: Iterator not enabled.")
            return (None, 0, )
        
        # store generator at start
        if self.index == 0:
            self.generator = generatorMaker()

        # iterate
        try:
            item = next(self.generator)
        except StopIteration:
            self.index = 0
            print(f"AG {self.NAME}: StopIteration encountered.")
            return (None, 0, )

        # get list item
        print(f"AG {self.NAME}: iterated {self.index + 1} times.")

        # -----------------------------------------------------
        # hack the prompt queue to add a new item to the queue
        # -----------------------------------------------------
        queue = PromptServer.instance.prompt_queue
        current_item = queue.currently_running[queue.task_counter - 1]

        # item structure: (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
        new_item = list(current_item) # copy current item
        new_item[0] = queue.task_counter # update index
        new_item[1] = str(uuid.uuid4()) # generate new prompt_id
        new_item[3]["create_time"] = int(time.time() * 1000)  # timestamp in milliseconds

        # add new item to queue
        queue.put(new_item)

        # return index
        current_idx = self.index
        self.index += 1

        return (item, current_idx, )
    
    @classmethod
    def IS_CHANGED(cls, generatorMaker, enabled, unique_id=None):
        if enabled:
            return float("nan") # enforce update
        return "unchanged" # don't update