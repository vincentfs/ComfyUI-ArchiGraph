CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/📀 OpenCV" # organized by py file

import torch
import numpy as np
import cv2
import comfy.model_management as model_management
from .anyType import ANY
from .enums import ColorEnum, ThreshEnum, BorderEnum, ContourModeEnum, \
    ContourMethodEnum, DepthEnum, ColormapEnum

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
        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            _, result[b] = cv2.threshold(nparrays[b], thresh, maxval, ThreshEnum[type])

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
        if ksize_y == 1: ksize_y = ksize_x

        batch_size = nparrays.shape[0]
        result = np.zeros_like(nparrays)

        for b in range(batch_size):
            result[b] = cv2.blur(nparrays[b], (ksize_x, ksize_y), borderType=BorderEnum[borderType])

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class Sobel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "dx": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Order of the derivative x."
                }),
                "dy": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Order of the derivative y."
                }),
                "ksize": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 31,
                    "step": 2, # Must be odd number
                    "tooltip": "Aperture size for the Sobel operator. Must be odd and greater than 1."
                }),
                "ddepth": (["CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F"], {
                    "default": "CV_32F",
                    "tooltip": "Desired depth of the destination image. Use CV_32F/64F for float output."
                }),
                "converted": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the output will be converted with cv2.convertScaleAbs() to 8-bit unsigned integer. If false, the output will remain unprocessed."
                }),
            },
        }
    
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Sobel"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Sobel. Computes the first, second, third, or mixed image derivatives using an extended Sobel operator."

    def execute(self, nparrays, dx, dy, ksize, ddepth, converted):
        batch_size = nparrays.shape[0]

        batches = []
        for b in range(batch_size):
            # force input to be gray
            if (len(nparrays[b].shape) == 3 and nparrays[b].shape[2] == 3):
                gray = cv2.cvtColor(nparrays[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = nparrays[b]

            if converted:
                batch = cv2.convertScaleAbs(cv2.Sobel(gray, DepthEnum[ddepth], dx, dy, ksize=ksize))
            else:
                batch = cv2.Sobel(gray, DepthEnum[ddepth], dx, dy, ksize=ksize)
            batches.append(batch)

        result = np.stack(batches, axis=0)
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
    
    RETURN_TYPES = ("LIST_LINE",)
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
    
    RETURN_TYPES = ("LIST_LINE",)
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
                    "default": "RETR_CCOMP",
                    "tooltip": "Contour retrieval mode"
                }),
                "method": (["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], {
                    "default": "CHAIN_APPROX_SIMPLE",
                    "tooltip": "Contour approximation method"
                }),
            },
        }
    
    SINGULAR_ONLY = True
    RETURN_TYPES = ("CV_CONTOURS", "CV_HIERARCHY", )
    RETURN_NAMES = ("contours", "hierarchy", )
    FUNCTION = "execute"
    NAME = "Find Contours"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Find Contours. Finds contours in a binary image. Default without offset.\n" \
    "Return: - Contours, each contour is stored as a vector of points.\n" \
    "Return: - hierarchy, containing info about topology. Same length as contours. Each has 4 elements, [next, prev, first child, parent]." 

    def execute(self, Np_bins, mode, method):
        if Np_bins.shape[0] != 1:
            raise ValueError("FindContours node only supports batch size of 1.")

        # Convert to grayscale if needed
        if len(Np_bins[0].shape) == 3 and Np_bins[0].shape[2] == 3:
            gray = cv2.cvtColor(Np_bins[0], cv2.COLOR_BGR2GRAY)
        else:
            gray = Np_bins[0]

        contours, hierarchy = cv2.findContours(gray, ContourModeEnum[mode], ContourMethodEnum[method])

        return (contours, hierarchy, )
    
#---------------------------------------------------------------------------------------------------------------------#
class DrawContours:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY", {
                    "tooltip": "Input image to draw contours on. Should be color image."
                }),
                "contours": ("CV_CONTOURS", {
                    "tooltip": "Contours to draw. Use FindContours node to get contours."
                }),
                "color": (["WHITE", "BLACK", "RED", "GREEN", "BLUE"], {
                    "default": "GREEN",
                    "tooltip": "color for contour."
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": -1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Thickness of the contour lines. -1 means filled."
                }),
                "hideBackground": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, the background will be blacked out, only contours are shown."
                }),
            },
        }
    
    SINGULAR_ONLY = True
    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Draw Contours"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Draw Contours. Always draws all contours in the list."

    def execute(self, nparrays, contours, color, thickness, hideBackground):
        if nparrays.shape[0] != 1:
            raise ValueError("DrawContours node only supports batch size of 1.")

        if hideBackground:
            height = nparrays.shape[1]
            width = nparrays.shape[2]
            input = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # convert to color if needed
            if len(nparrays[0].shape) == 2 or nparrays[0].shape[2] == 1:
                input = cv2.cvtColor(nparrays[0], cv2.COLOR_GRAY2BGR)
            else:
                input = nparrays[0]
        
        result = cv2.drawContours(input, contours, -1, ColorEnum[color], thickness)
        return (result[np.newaxis, :],)

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

        result = np.zeros((batch_size, nparrays.shape[1], nparrays.shape[2], 3), dtype=np.uint8)

        for b in range(batch_size):
            # Ensure the image is color
            if len(nparrays[b].shape) == 2 or nparrays[b].shape[2] == 1:
                result[b] = cv2.cvtColor(nparrays[b], cv2.COLOR_GRAY2BGR)
            else:
                result[b] = nparrays[b]

            for point in points[b]:
                result[b] = cv2.circle(result[b], point, radius, ColorEnum[color], thickness)

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
                "removeSurrounding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the surrounding part (different from background) will be removed from output."
                }),
            },
        }
    
    SINGULAR_ONLY = True
    RETURN_TYPES = ("INT", "NPARRAY", ANY, ANY,)
    RETURN_NAMES = ("num_labels", "labels", "stats", "centroids",)
    FUNCTION = "execute"
    NAME = "Connected Components With Stats"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "OpenCV Connected Components With Stats. Labels connected components in a binary image and computes statistics for each component."

    def execute(self, Np_bin, connectivity, removeSurrounding):
        if Np_bin.shape[0] != 1:
            raise ValueError("ConnectedComponentsWithStats node only supports batch size of 1.")
        ConnectivityInput = 4 if connectivity == "4" else 8

        # Convert to grayscale if needed
        if len(Np_bin[0].shape) == 3 and Np_bin[0].shape[2] == 3:
            gray = cv2.cvtColor(Np_bin[0], cv2.COLOR_BGR2GRAY)
        else:
            gray = Np_bin[0]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=ConnectivityInput)

        if num_labels <= 1:
            raise ValueError("No objects found in the binary image.")
        
        if removeSurrounding:
            corner_labels = (labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1])
            # if more than 3 corners share the same label, consider it as surrounding
            surrounding_label = max(set(corner_labels), key=corner_labels.count)
            if surrounding_label != 0 and corner_labels.count(surrounding_label) >= 3:
                labels[labels == surrounding_label] = 0  # set surrounding to background
                # re-label the largest label to use surrounding_label
                labels[labels == num_labels - 1] = surrounding_label
                num_labels -= 1

        return (num_labels, labels, stats, centroids, )