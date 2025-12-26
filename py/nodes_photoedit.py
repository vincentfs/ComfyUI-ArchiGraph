CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/📷 PhotoEdit" # organized by py file

import torch
from comfy.utils import common_upscale
from ..node_registry import comfy_node

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
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
                "target_size": (["720P", "1080P", "2K", "4K", "custom"], {
                    "default": "720P", # Default value
                    "tooltip": "Preset target sizes. Select 'custom' to specify edge size manually.",
                }),
                "bbox_edge1": ("INT", {
                    "default": 1024, 
                    "min": 16, #Minimum value
                    "max": 8192, #Maximum value
                    "step": 2, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "tooltip": "Only valid for custom target type. Edge size for the boundingbox. Orientation will be calculated automatically.",
                }),
                "bbox_edge2": ("INT", {
                    "default": 1024, 
                    "min": 16, #Minimum value
                    "max": 8192, #Maximum value
                    "step": 2, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "tooltip": "Only valid for custom target type. Edge size for the boundingbox. Orientation will be calculated automatically.",
                }),
                "upscale_method": (cls.upscale_methods,),
                "onlyEnlarge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, only enlarge images. Large images will be kept at original size.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "prev_width", "prev_height", "log")
    FUNCTION = "resize"
    NAME = "Image Resize To"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Resizes an image by specify one edge size. Proportion is kept by default."

    def resize(self, image: torch.Tensor, target_size: str, bbox_edge1: int, bbox_edge2: int, upscale_method: str, onlyEnlarge: bool):

        # get original image size
        B, H, W, C = image.shape
        prev_width = W
        prev_height = H

        image = image.movedim(-1, 1)

        ratio = max(W, H) * 1.0 / min(W, H)
        short_edge = 0
        long_edge = 0

        if target_size == "custom":
            bbox_short = min(bbox_edge1, bbox_edge2)
            bbox_long = max(bbox_edge1, bbox_edge2)
            new_ratio = bbox_long * 1.0 / bbox_short
            if ratio > new_ratio:
                short_edge = bbox_short
                long_edge = int(bbox_short * ratio)
            else:
                long_edge = bbox_long
                short_edge = int(bbox_long / ratio)
        else:
            new_ratio = 16.0 / 9.0
            if ratio > new_ratio:
                short_edge = 720
                long_edge = int(720 * ratio)
            else:
                long_edge = 1280
                short_edge = int(1280 / ratio)

            if target_size == "720P":
                pass
            elif target_size == "1080P":
                short_edge *= 1.5
                long_edge *= 1.5
            elif target_size == "2K":
                short_edge *= 2
                long_edge *= 2
            elif target_size == "4K":
                short_edge *= 3
                long_edge *= 3

        if W >= H:
            new_width = long_edge
            new_height = short_edge
        else:
            new_width = short_edge
            new_height = long_edge

        if onlyEnlarge and new_width <= W:
            log = f"Original image is larger, not resized."
        else:
            image = common_upscale(image, new_width, new_height, upscale_method, "disabled")
            log = f"Resized from W{prev_width}xH{prev_height} to W{new_width}xH{new_height}."

        image = image.movedim(1, -1)
        return (image, prev_width, prev_height, log,)

#---------------------------------------------------------------------------------------------------------------------#
@comfy_node
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

#---------------------------------------------------------------------------------------------------------------------#
