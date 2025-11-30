CURRENT_CATEGORY = "🐦‍🔥 ArchiGraph/✨ Xfeat" # organized by py file

import math
import numpy as np
import numba as nb
import cv2
from .anyType import ANY

#---------------------------------------------------------------------------------------------------------------------#
@nb.njit(parallel=True, fastmath=True)
def getAccumulator(points: list[tuple[int, int]], ticks: tuple[np.ndarray, np.ndarray], \
                   rho_lower: float, rho_upper: float, rho_res: float, \
                    thetas_rad: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    '''
    Must be outside of class to be compatible with numba JIT. Otherwise, numba will throw an error, because it cannot compile class methods directly.

    Compute the Hough accumulator for a set of points.
    Args:
        points (list of tuple): List of (y, x) coordinates of edge points.
        ticks (tuple of list): Tuple containing two lists - [0]rho_ticks and [1]theta_ticks.
        ...
    Returns:
        np.ndarray: Hough accumulator array of shape (num_rho_bins, num_theta_bins).
    '''
    # Create accumulator
    acc = np.zeros((len(ticks[0]), len(ticks[1])), dtype=np.uint16)
    pointCount = len(points)

    for i in nb.prange(pointCount): 
        y = points[i][0]
        x = points[i][1]
        
        for j in range(len(thetas_rad)):
            rho = x * cos_t[j] + y * sin_t[j]

            if rho < rho_lower or rho > rho_upper:
                continue  # Skip if rho is out of bounds

            # map rho to bin index
            rho_bin = int((rho - rho_lower) / rho_res)
            acc[rho_bin, j] += 1

    return acc

#---------------------------------------------------------------------------------------------------------------------#
class HoughLinesAccumulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "np_bins": ("NPARRAY",),
                "theta_res": ("FLOAT", {
                    "default": 1,
                    "min": 0.05,
                    "max": 2,
                    "step": 0.05,
                    "tooltip": "Angular resolution in degrees.",
                }),
                "theta_bnd1": ("FLOAT", {
                    "default": -90,
                    "min": -90,
                    "max": 90,
                    "step": 0.1,
                    "tooltip": "Range boundary of theta.",
                }),
                "theta_bnd2": ("FLOAT", {
                    "default": 90,
                    "min": -90,
                    "max": 90,
                    "step": 0.1,
                    "tooltip": "Range boundary of theta.",
                }),
                "rho_res": ("FLOAT", {
                    "default": 1,
                    "min": 0.5,
                    "max": 2,
                    "step": 0.1,
                    "tooltip": "Distance resolution in pixels.",
                }),
                "rho_bnd1": ("FLOAT", {
                    "default": -1,
                    "min": -1,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Range boundary of rho as percentage.",
                }),
                "rho_bnd2": ("FLOAT", {
                    "default": 1,
                    "min": -1,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Range boundary of rho as percentage.",
                }),
            },
        }

    RETURN_TYPES = ("NPARRAY", ANY)
    RETURN_NAMES = ("dst", "ticks")
    FUNCTION = "execute"
    NAME = "Hough Lines Accumulator"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Accumulates Hough lines from a batch of binary images. Outputs the accumulator array with dtype uint16."

    def execute(self, np_bins, theta_res, theta_bnd1, theta_bnd2, rho_res, rho_bnd1, rho_bnd2):
        batch_size = np_bins.shape[0]

        # 1. Contruct Parameter Space
        H, W = np_bins.shape[1:3]

        # Generate theta values within specified bounds
        theta_lower = min(theta_bnd1, theta_bnd2)
        theta_upper = max(theta_bnd1, theta_bnd2)
        theta_ticks = np.arange(theta_lower, theta_upper, theta_res)
        if len(theta_ticks) == 0:
            # Fallback to default range if no theta ticks are generated
            theta_ticks = np.arange(-90, 90, theta_res)

        # Generate rho values within specified bounds
        diag = np.hypot(H, W) # max possible rho value
        rho_lower = diag * min(rho_bnd1, rho_bnd2)
        rho_upper = diag * max(rho_bnd1, rho_bnd2)
        rho_tickCount = math.ceil((rho_upper - rho_lower)/rho_res) 
        if rho_tickCount <= 0:
            # Fallback to default range if no rho ticks are generated
            rho_tickCount = math.ceil(2 * diag/rho_res)
            rho_lower = -diag
        rho_ticks = np.fromiter((rho_lower + r * rho_res for r in range(rho_tickCount)), dtype=np.float32)

        # Precompute values for efficiency
        thetas_rad = np.deg2rad(theta_ticks)
        cos_t = np.cos(thetas_rad)
        sin_t = np.sin(thetas_rad)

        # Initialize result accumulator
        result = np.zeros((batch_size, len(rho_ticks), len(theta_ticks)), dtype=np.uint16)
        ticks = (rho_ticks, theta_ticks)
        ticks_dict = {"rho_ticks": rho_ticks, "theta_ticks": theta_ticks} # for easier access in downstream nodes

        # 2. Accumulate Votes
        for b in range(batch_size):
            # Convert to grayscale if needed
            if len(np_bins[b].shape) == 3 and np_bins[b].shape[2] == 3:
                gray = cv2.cvtColor(np_bins[b], cv2.COLOR_BGR2GRAY)
            else:
                gray = np_bins[b]

            # Get coordinates of edge pixels
            y_idx, x_idx = np.nonzero(gray) 
            points = list(zip(y_idx, x_idx))

            result[b] = getAccumulator(points, ticks, rho_lower, rho_upper, rho_res, thetas_rad, cos_t, sin_t)

        return (result, ticks_dict, )

class HoughLinesPlotter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "lines": (ANY, {
                    "default": None,
                    "tooltip": "lines represented as (rho, theta_in_rad) or (x1, y1, x2, y2) tuples.",
                }),
                "color": (["WHITE", "BLACK", "RED", "GREEN", "BLUE"], {
                    "default": "GREEN",
                    "tooltip": "Line Color."
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Line thickness."
                }),
            },
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("dst",)
    FUNCTION = "execute"
    NAME = "Hough Lines Plotter"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Plots detected Hough lines onto base images. Outputs color images with lines drawn."

    def execute(self, nparrays, lines, color, thickness):
        batch_size = nparrays.shape[0]
        colorEnum = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "RED":   (0, 0, 255),
            "GREEN": (0, 255, 0),
            "BLUE":  (255, 0, 0),
        }

        result = np.zeros((batch_size, nparrays.shape[1], nparrays.shape[2], 3), dtype=np.uint8)

        scale = max(nparrays.shape[1], nparrays.shape[2]) * 2 # extend length for drawing lines

        for b in range(batch_size):
            # Ensure the image is color
            if len(nparrays[b].shape) == 2 or nparrays[b].shape[2] == 1:
                result[b] = cv2.cvtColor(nparrays[b], cv2.COLOR_GRAY2BGR)
            else:
                result[b] = nparrays[b]

            if len(lines[b]) == 0: continue # skip if no lines detected

            print(lines[b][0])

            # different line formats
            if len(lines[b][0]) == 4:
                # lines as (x1, y1, x2, y2)
                for x1, y1, x2, y2 in lines[b]:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    result[b] = cv2.line(result[b], pt1, pt2, colorEnum[color], thickness)
            elif len(lines[b][0]) == 2:
                # lines as (rho, theta)
                for rho, theta in lines[b]:
                    cos = np.cos(theta)
                    sin = np.sin(theta)
                    # Foot of perpendicular from origin to the line
                    x0, y0 = cos * rho, sin * rho
                    # Direction vector
                    dx, dy = -sin, cos
                    # extend two directions
                    pt1 = (int(x0 + dx * scale), int(y0 + dy * scale))
                    pt2 = (int(x0 - dx * scale), int(y0 - dy * scale))

                    result[b] = cv2.line(result[b], pt1, pt2, colorEnum[color], thickness)

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class KLargestPoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nparrays": ("NPARRAY",),
                "k": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of largest points to extract.",
                }),
            },
            "optional": {
                "ticks": (ANY, {
                    "default": None,
                    "tooltip": "Optional ticks input to remap points.",
                }),
                
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("points",)
    FUNCTION = "execute"
    NAME = "K Largest Points"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Extracts the K largest values from each 2D numpy array in the batch. Outputs each batch's largest values as a size k list of (x, y) coordinates."

    def execute(self, nparrays, k, ticks=None):
        batch_size = nparrays.shape[0]

        if nparrays.ndim != 3:
            raise ValueError("Input nparrays must be a batch of 2D arrays.")
        
        result = []

        for b in range(batch_size):
            input = nparrays[b]
            points = []
            # Flatten the array and get the indices of the k largest values
            flat_indices = np.argpartition(input.flatten(), -k)[-k:]
            # Convert flat indices back to 2D coordinates
            for idx in flat_indices:
                x = idx % input.shape[1]
                y = idx // input.shape[1]
                if ticks is not None:
                    # Remap points using ticks if provided
                    y = ticks["rho_ticks"][y]
                    x = ticks["theta_ticks"][x]
                points.append((x, y))

            result.append(points)

        return (result,)

#---------------------------------------------------------------------------------------------------------------------#
class PointToLineParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": (ANY, {
                    "default": None,
                    "tooltip": "Points as (x, y) or (theta in degrees, rho) tuples.",
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("lines",)
    FUNCTION = "execute"
    NAME = "Point to Line Params"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Converts (x, y) points to Hough line parameters (rho, theta in rads). Outputs a list of (rho, theta) tuples."

    def execute(self, points):
        result = []

        for batch_points in points:
            lines = []
            for point in batch_points:
                rho = point[1]
                theta_rad = np.deg2rad(point[0])
                lines.append((rho, theta_rad))
            result.append(lines)

        return (result,)

class ColormapHotGray:
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
    NAME = "Colormap Hot Gray"
    CATEGORY = CURRENT_CATEGORY
    DESCRIPTION = "Maps a numpy array to a gray colormap for visualization. Outputs a single channel gray image."

    def execute(self, nparrays):
        batch_size = nparrays.shape[0]

        if nparrays.ndim != 3:
            raise ValueError("Input nparrays must be a batch of 2D arrays.")
        
        result = np.zeros_like(nparrays, dtype=np.uint8)

        for b in range(batch_size):
            input = nparrays[b]
            # Normalize the array to 0-255
            arr_min = np.min(input)
            arr_max = np.max(input)
            base = max(arr_max,1)
            result[b] = ((input - arr_min) / base * 255).astype(np.uint8)

        return (result,)