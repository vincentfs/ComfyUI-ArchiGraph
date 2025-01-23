import torch

from ..categories import icons
from comfy.utils import common_upscale

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
    NAME = "AG Image Resize To"
    CATEGORY = icons.get(NAME)
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

        return (image, prev_width, prev_height,log,)

#---------------------------------------------------------------------------------------------------------------------#
class Image2BW:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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
    NAME = "AG Image to BW"
    CATEGORY = icons.get(NAME)
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