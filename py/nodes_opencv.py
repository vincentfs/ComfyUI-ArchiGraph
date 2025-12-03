CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/⭕ OpenCV" # organized by py file

import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .anyType import ANY

#---------------------------------------------------------------------------------------------------------------------#
# Helpers
#---------------------------------------------------------------------------------------------------------------------#
class Image2Nparray:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK", {
                    "default": None,
                    "tooltip": "Optional mask to convert to Nparray. If provided, will overwrite the images input."
                })
            }
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("nparrays",)
    FUNCTION = "execute"
    NAME = "To Nparray"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Convert images to Nparrays for use by OpenCV. Supports batch size >= 1."
    
    def execute(self, images=None, masks=None):
        if masks is not None:
            # Convert From Torch Tensor [B, H, W] to Numpy Array [B, H, W, 1]. Clip [0,1] as sanity check.
            ret = (masks.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)[..., np.newaxis]
        elif images is not None:
            # Convert From Torch Tensor [B, H, W, C] to Numpy Array. Clip [0,1] as sanity check.
            ret = (images.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)[..., ::-1] # reverse color channel from RGB to BGR
        else:
            ret = np.zeros((1,1,1,1),dtype=np.uint8)  # return dummy value if both inputs are None
        return (ret,)

#---------------------------------------------------------------------------------------------------------------------#
class Nparray2Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    NAME = "To Image"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Convert OpenCV Nparrays to images for preview. Supports batch size >= 1."
    
    def execute(self, nparrays):
        images_hwc = []
        for nparray in nparrays:
            if	len(nparray.shape)	== 2: nparray = cv2.cvtColor(nparray, cv2.COLOR_GRAY2RGB	)	# Grayscale image (H, W)
            elif	nparray.shape[2]	== 1: nparray = cv2.cvtColor(nparray, cv2.COLOR_GRAY2RGB	)	# Single-channel grayscale (H, W, 1)
            elif	nparray.shape[2]	== 3: nparray = cv2.cvtColor(nparray, cv2.COLOR_BGR2RGB	)	# BGR image (H, W, 3)

            image_hwc = torch.from_numpy(nparray.astype(np.float32) / 255.0).to(model_management.get_torch_device())
            images_hwc.append(image_hwc)
        ret = torch.stack(images_hwc)
        return (ret,)

#---------------------------------------------------------------------------------------------------------------------#
# OpenCV Image Processing Nodes
#---------------------------------------------------------------------------------------------------------------------#
class Bitwise_not:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
            },
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Bitwise Not"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Bitwise Not. Inverts every bit of an array."

    def execute(self, nparrays):
        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            result[b] = cv2.bitwise_not(nparrays[b])

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class Threshold:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "thresh": ("FLOAT", {
                    "default": 127.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "tooltip": "Threshold value."
                }),
                "maxval": ("FLOAT", {
                    "default": 255.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "tooltip": "Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV thresholding types."
                }),
                "type": (["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"], {
                    "default": "BINARY",
                    "tooltip": "Thresholding type."
                }),
            },
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Threshold"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Threshold. Applies a fixed-level threshold to each array element."

    def execute(self, nparrays, thresh, maxval, type):
        TypeEnum = {
            "BINARY": cv2.THRESH_BINARY,
            "BINARY_INV": cv2.THRESH_BINARY_INV,
            "TRUNC": cv2.THRESH_TRUNC,
            "TOZERO": cv2.THRESH_TOZERO,
            "TOZERO_INV": cv2.THRESH_TOZERO_INV,
        }

        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            _, result[b] = cv2.threshold(nparrays[b], thresh, maxval, TypeEnum[type])

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class GaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "ksize_x": ("INT", {
                    "default": 3,
                    "min": 3,
                    "max": 99,
                    "step": 2, # Must be odd number
                    "tooltip": "Kernel width. Must be odd and greater than 1."
                }),
                "ksize_y": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 99,
                    "step": 2, # Must be odd number
                    "tooltip": "Kernel height. Must be odd and greater than 1. Set to 1 to use same as width."
                }),
                "sigma_x": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "Gaussian kernel standard deviation in X direction."
                }),
                "sigma_y": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively."
                }),
                "borderType": (["DEFAULT", "CONSTANT", "REPLICATE", "WRAP"], {
                    "default": "DEFAULT",
                    "tooltip": "Pixel extrapolation method",
                }),
            },
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Blur Gaussian"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV GaussianBlur. Default anchor is center of the kernel."

    def execute(self, nparrays, ksize_x, ksize_y, sigma_x, sigma_y, borderType):
        BorderEnum = {
            "DEFAULT": cv2.BORDER_DEFAULT,
            "CONSTANT": cv2.BORDER_CONSTANT,
            "REPLICATE": cv2.BORDER_REPLICATE,
            "WRAP": cv2.BORDER_WRAP,
        }
        if ksize_y == 1: ksize_y = ksize_x

        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            result[b] = cv2.GaussianBlur(nparrays[b], (ksize_x, ksize_y), sigma_x, sigmaY=sigma_y, borderType=BorderEnum[borderType])

        return (result,)
    
#---------------------------------------------------------------------------------------------------------------------#
class MedianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "ksize": ("INT", {
                    "default": 3,
                    "min": 3,
                    "max": 99,
                    "step": 2, # Must be odd number
                    "tooltip": "Aperture linear size; it must be odd and greater than 1, e.g. 3, 5, 7 ..."
                }),
            },
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Blur Median"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV MedianBlur."

    def execute(self, nparrays, ksize):
        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            result[b] = cv2.medianBlur(nparrays[b], ksize)

        return (result,)
    
#---------------------------------------------------------------------------------------------------------------------#
class Blur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "ksize_x": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 99,
                    "step": 2, # Must be odd number
                    "display": "slider",
                    "tooltip": "Kernel width. Must be odd and greater than 1."
                }),
                "ksize_y": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 99,
                    "step": 2, # Must be odd number
                    "display": "slider",
                    "tooltip": "Kernel height. Must be odd and greater than 1. Set to 1 to use same as width."
                }),
                "borderType": (["DEFAULT", "CONSTANT", "REPLICATE", "WRAP"], {
                    "default": "DEFAULT",
                    "tooltip": "Pixel extrapolation method",
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Blur"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Blur. Default anchor is center of the kernel."

    def execute(self, nparrays, ksize_x, ksize_y, borderType):
        BorderEnum = {
            "DEFAULT": cv2.BORDER_DEFAULT,
            "CONSTANT": cv2.BORDER_CONSTANT,
            "REPLICATE": cv2.BORDER_REPLICATE,
            "WRAP": cv2.BORDER_WRAP,
        }
        if ksize_y == 1: ksize_y = ksize_x

        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            result[b] = cv2.blur(nparrays[b], (ksize_x, ksize_y), borderType=BorderEnum[borderType])

        return (result,)
    
#---------------------------------------------------------------------------------------------------------------------#
class Canny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "threshold1": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "First threshold for the hysteresis procedure."
                }),
                "threshold2": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Second threshold for the hysteresis procedure."
                }),
                "apertureSize": ("INT", {
                    "default": 3,
                    "min": 3,
                    "max": 7,
                    "step": 2, # Must be odd number
                    "display": "slider",
                    "tooltip": "Aperture size for the Sobel operator."
                }),
                "L2gradient": ("BOOLEAN", {
                    "default": True, # Default value
                    "tooltip": "A flag, indicating whether a more accurate L2 norm should be used to calculate the image gradient magnitude."
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("Np_grays",)
    FUNCTION = "execute"
    NAME = "Canny"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Canny Edge Detection. Usually requires Gaussian blur before running. Runs on CPU, different from ComfyUI Kornia version which runs on GPU.\n" \
    "Return Shape [B,H,W]."

    def execute(self, nparrays, threshold1, threshold2, apertureSize, L2gradient):
        batch_size = nparrays.shape[0]
        height = nparrays.shape[1]
        width = nparrays.shape[2]
        # create a properly shaped uint8 array for the results: (batch, H, W)
        result = np.zeros((batch_size, height, width), dtype=np.uint8)

        for b in range(batch_size):
            # Convert to grayscale if needed
            if len(nparrays[b].shape) == 3 and nparrays[b].shape[2] == 3:
                gray = cv2.cvtColor(nparrays[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = nparrays[b]

            edges = cv2.Canny(gray,
                              threshold1=threshold1 * 255,
                              threshold2=threshold2 * 255,
                              apertureSize=apertureSize,
                              L2gradient=L2gradient)
            result[b] = edges

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class HoughLines:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {
                    "tooltip": "8-bit, single-channel binary source image."
                }),
                "rho_res": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2,
                    "step": 0.1,
                    "tooltip": "Distance resolution of the accumulator in pixels."
                }),
                "theta_res": ("FLOAT", {
                    "default": 1,
                    "min": 0.05,
                    "max": 2,
                    "step": 0.05,
                    "tooltip": "Angle resolution of the accumulator in degrees."
                }),
                "threshold": ("INT", {
                    "default": 100,
                    "min": 1,
                    "tooltip": "Accumulator threshold parameter. Only those lines are returned that get enough votes (> threshold)."
                }),
            },
        }
    
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("lines",)
    FUNCTION = "execute"
    NAME = "Hough Lines"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Hough Line Transform. Detects lines in a binary image using the standard Hough Transform."

    def execute(self, nparrays, rho_res, theta_res, threshold):
        batch_size = nparrays.shape[0]
        list_lines = []

        for b in range(batch_size):
            # Convert to grayscale if needed
            if len(nparrays[b].shape) == 3 and nparrays[b].shape[2] == 3:
                gray = cv2.cvtColor(nparrays[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = nparrays[b]

            # default lines is (N,1,2) array of (rho, theta)
            lines = cv2.HoughLines(gray,
                                   rho=rho_res,
                                   theta=np.deg2rad(theta_res),
                                   threshold=threshold)
            if lines is None:
                lines = []
            else:
                lines = lines[:, 0, :].tolist()  # Convert to list of (rho, theta)

            list_lines.append(lines)

        return (list_lines,)

#---------------------------------------------------------------------------------------------------------------------#
class HoughLinesP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {
                    "tooltip": "8-bit, single-channel binary source image."
                }),
                "rho_res": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2,
                    "step": 0.1,
                    "tooltip": "Distance resolution of the accumulator in pixels."
                }),
                "theta_res": ("FLOAT", {
                    "default": 1,
                    "min": 0.05,
                    "max": 2,
                    "step": 0.05,
                    "tooltip": "Angle resolution of the accumulator in degrees."
                }),
                "threshold": ("INT", {
                    "default": 100,
                    "min": 1,
                    "tooltip": "Accumulator threshold parameter. Only those lines are returned that get enough votes (> threshold)."
                }),
            },
        }
    
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("lines",)
    FUNCTION = "execute"
    NAME = "Hough Lines Probabilistic"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Probabilistic Hough Line Transform. Detects line segments in a binary image."

    def execute(self, nparrays, rho_res, theta_res, threshold):
        batch_size = nparrays.shape[0]
        list_lines = []

        for b in range(batch_size):
            # Convert to grayscale if needed
            if len(nparrays[b].shape) == 3 and nparrays[b].shape[2] == 3:
                gray = cv2.cvtColor(nparrays[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = nparrays[b]

            # default lines is (N,1,4) array of (x1, y1, x2, y2)
            lines = cv2.HoughLinesP(gray, rho_res, np.deg2rad(theta_res), threshold)
            if lines is None:
                lines = []
            else:
                lines = lines[:, 0, :].tolist()  # Convert to list of (x1, y1, x2, y2)

            list_lines.append(lines)

        return (list_lines,)

#---------------------------------------------------------------------------------------------------------------------#
class FindContours:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Np_bins": ("NPARRAY", {
                    "tooltip": "Binary image. Non-zero pixels are treated as 1's. \nYou can use compare(), threshold(), inRange(), Canny() etc. to create a binary image."
                }),
                "mode": (["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE"], {
                    "default": "RETR_EXTERNAL",
                    "tooltip": "Contour retrieval mode"
                }),
                "method": (["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], {
                    "default": "CHAIN_APPROX_SIMPLE",
                    "tooltip": "Contour approximation method"
                }),
            },
        }
    
    RETURN_TYPES = (ANY, ANY, )
    RETURN_NAMES = ("contours", "hierachy", )
    FUNCTION = "execute"
    NAME = "Find Contours"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Find Contours. Finds contours in a binary image. Default without offset.\n" \
    "Return: - Contours, each contour is stored as a vector of points.\n" \
    "Return: - hierarchy, containing info about topology. Same length as contours. Each has 4 elements, [next, prev, first child, parent]." 

    def execute(self, Np_bins, mode, method):
        ModeEnum = {
            "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
            "RETR_LIST": cv2.RETR_LIST,
            "RETR_CCOMP": cv2.RETR_CCOMP,
            "RETR_TREE": cv2.RETR_TREE,
        }
        MethodEnum = {
            "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
            "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
            "CHAIN_APPROX_TC89_L1": cv2.CHAIN_APPROX_TC89_L1,
            "CHAIN_APPROX_TC89_KCOS": cv2.CHAIN_APPROX_TC89_KCOS,
        }

        batch_size = Np_bins.shape[0]

        list_contours = []
        list_hierarchy = []

        for b in range(batch_size):
            # Convert to grayscale if needed
            if len(Np_bins[b].shape) == 3 and Np_bins[b].shape[2] == 3:
                gray = cv2.cvtColor(Np_bins[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = Np_bins[b]

            contours, hierarchy = cv2.findContours(gray, ModeEnum[mode], MethodEnum[method])
            list_contours.append(contours)
            list_hierarchy.append(hierarchy)

            # contour_image = np.zeros_like(Np_bins[b])
            # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        return (list_contours, list_hierarchy, )
    
#---------------------------------------------------------------------------------------------------------------------#
class DrawContours:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {
                    "tooltip": "Input image to draw contours on. Should be color image."
                }),
                "contours": (ANY, {
                    "tooltip": "Contours to draw. Use FindContours node to get contours."
                }),
                "color": (["WHITE", "BLACK", "RED", "GREEN", "BLUE"], {
                    "default": "GREEN",
                    "tooltip": "color for contour."
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Thickness of the contour lines."
                }),
                "hideBackground": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, the background will be blacked out, only contours are shown."
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Draw Contours"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Draw Contours. Always draws all contours in the list."

    def execute(self, nparrays, contours, color, thickness, hideBackground):
        batch_size = min(nparrays.shape[0], len(contours)) # in case different batch size, use the smaller ones
        colorEnum = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "RED":   (0, 0, 255),
            "GREEN": (0, 255, 0),
            "BLUE":  (255, 0, 0),
        }

        result = np.zeros_like(nparrays)

        if hideBackground:
            height = nparrays.shape[1]
            width = nparrays.shape[2]
            input = np.zeros((height, width, 3), dtype=np.uint8)
            for b in range(batch_size):
                result[b] = cv2.drawContours(input, contours[b], -1, colorEnum[color], thickness)
        else:
            for b in range(batch_size):
                result[b] = cv2.drawContours(nparrays[b], contours[b], -1, colorEnum[color], thickness)

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class Circles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {
                    "tooltip": "Input image to draw circles on. Will be convert to color image."
                }),
                "points": (ANY, {
                    "tooltip": "list of center points for the circles. Shape should be (batch_size, num_circles, 2) where last dimension is (x, y)."
                }),
                "radius": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Radius of the circles."
                }),
                "color": (["WHITE", "BLACK", "RED", "GREEN", "BLUE"], {
                    "default": "RED",
                    "tooltip": "Color of the circles."
                }),
                "thickness": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Thickness of the circle outlines, if positive. Negative thickness means filled circle."
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Draw Circles"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Draw Circles. Draws multiple circles on each image based on provided center points."

    def execute(self, nparrays, points, radius, color, thickness):
        batch_size = nparrays.shape[0]
        colorEnum = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "RED":   (0, 0, 255),
            "GREEN": (0, 255, 0),
            "BLUE":  (255, 0, 0),
        }

        result = np.zeros((batch_size, nparrays.shape[1], nparrays.shape[2], 3), dtype=np.uint8)

        for b in range(batch_size):
            # Ensure the image is color
            if len(nparrays[b].shape) == 2 or nparrays[b].shape[2] == 1:
                result[b] = cv2.cvtColor(nparrays[b], cv2.COLOR_GRAY2BGR)
            else:
                result[b] = nparrays[b]

            for point in points[b]:
                result[b] = cv2.circle(result[b], point, radius, colorEnum[color], thickness)

        return (result,)
    
#---------------------------------------------------------------------------------------------------------------------#
class ConnectedComponentsWithStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Np_bin": ("NPARRAY", {
                    "tooltip": "Single batch binary image. Non-zero pixels are treated as 1's. \nYou can use compare(), threshold(), inRange(), Canny() etc. to create a binary image."
                }),
                "connectivity": (["4", "8"], {
                    "default": "8",
                    "tooltip": "Pixel connectivity to use."
                }),
                "colormap": (list(cls._COLORMAPS.keys()), {
                    "default": "HSV",
                    "tooltip": "Colormap to use for preview."
                }),
            },
        }
    
    SINGULAR_ONLY = True
    RETURN_TYPES = ("INT", "NPARRAY", ANY, ANY, "NPARRAY",)
    RETURN_NAMES = ("num_labels", "labels", "stats", "centroids", "preview",)
    FUNCTION = "execute"
    NAME = "Connected Components With Stats"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Connected Components With Stats. Labels connected components in a binary image and computes statistics for each component."

    _COLORMAPS = {
        'AUTUMN': cv2.COLORMAP_AUTUMN,
        'BONE': cv2.COLORMAP_BONE,
        'JET': cv2.COLORMAP_JET,
        'WINTER': cv2.COLORMAP_WINTER,
        'RAINBOW': cv2.COLORMAP_RAINBOW,  # 真正的彩虹色阶
        'OCEAN': cv2.COLORMAP_OCEAN,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'SPRING': cv2.COLORMAP_SPRING,
        'COOL': cv2.COLORMAP_COOL,
        'HSV': cv2.COLORMAP_HSV,
        'PINK': cv2.COLORMAP_PINK,
        'HOT': cv2.COLORMAP_HOT,
        'PARULA': cv2.COLORMAP_PARULA,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'CIVIDIS': cv2.COLORMAP_CIVIDIS,
        'TWILIGHT': cv2.COLORMAP_TWILIGHT,
        'TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,
        'TURBO': cv2.COLORMAP_TURBO,
        'DEEPGREEN': cv2.COLORMAP_DEEPGREEN,
    }

    def execute(self, Np_bin, connectivity, colormap):
        if Np_bin.shape[0] != 1:
            raise ValueError("ConnectedComponentsWithStats node only supports batch size of 1.")
        ConnectivityInput = 4 if connectivity == "4" else 8

        # Convert to grayscale if needed
        if len(Np_bin.shape) == 3 and Np_bin.shape[2] == 3:
            gray = cv2.cvtColor(Np_bin[0], cv2.COLOR_BGR2GRAY)
        else:
            gray = Np_bin[0]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=ConnectivityInput)

        if num_labels <= 1:
            raise ValueError("No objects found in the binary image.")
        elif num_labels == 2:
            # use black and white directly
            preview = (labels * 255).astype(np.uint8)[np.newaxis, :]
        else:
            # normalize the labels to 0-255 range for visualization
            num_objects = num_labels - 1  # Exclude background label 0
            normalized_labels = labels * (255 // num_objects)  # Normalize to 0-255, will never reach 1
            colored = cv2.applyColorMap(normalized_labels.astype(np.uint8), self._COLORMAPS[colormap])
            # force label=0 to be black, because colormap[0] is usually not pure black (like deep blue/purple)
            colored[labels == 0] = [0, 0, 0]
            preview = colored[np.newaxis, :]


        return (num_labels, labels, stats, centroids, preview,)