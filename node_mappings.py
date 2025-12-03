from .py.log import log
import importlib
# import importlib.util # for auto import

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def AddMapping(node_class):
    # Add AG prefix to original name for avoid conflict
    NODE_CLASS_MAPPINGS["AG "+node_class.NAME] = node_class

    # check whether the node has attribute SINGULAR_ONLY
    if not hasattr(node_class, "SINGULAR_ONLY") or not node_class.SINGULAR_ONLY:
        NODE_DISPLAY_NAME_MAPPINGS["AG "+node_class.NAME] = node_class.NAME+" 🐦‍🔥"
    else:
        NODE_DISPLAY_NAME_MAPPINGS["AG "+node_class.NAME] = node_class.NAME+" 🐦"

#--------------------------------------------------------------------------------#
# Load Image Nodes
#--------------------------------------------------------------------------------#
try:
    nodes_utils = importlib.import_module(".py.nodes_utils", package=__package__)
    for name in (
        "JsonSave",
        "JsonLoad",
        "NparraySave",
        "NparrayLoad",

        "Printf",
        "SliceList",
        "SliceSublist",

        "ImageResizeTo",
        "Image2BW",
    ):
        if hasattr(nodes_utils, name):
            AddMapping(getattr(nodes_utils, name))

except ImportError as e:
    log("Failed to load util nodes", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")

#--------------------------------------------------------------------------------#
# Load OpenCV Nodes
#--------------------------------------------------------------------------------#
try:
    nodes_opencv = importlib.import_module(".py.nodes_opencv", package=__package__)
    for name in (
        "Image2Nparray",
        "Nparray2Image",

        "Bitwise_not",
        "Threshold",
        "GaussianBlur",
        "MedianBlur",
        "Blur",

        "Canny",
        "HoughLines",
        "HoughLinesP",
        "FindContours",
        "DrawContours",
        "Circles",

        "ConnectedComponentsWithStats",
    ):
        if hasattr(nodes_opencv, name):
            AddMapping(getattr(nodes_opencv, name))

except ImportError as e:
    log("Failed to load opencv nodes", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")

#--------------------------------------------------------------------------------#
# Load XFeat Nodes
#--------------------------------------------------------------------------------#
try:
    # use a dynamic (string) import to avoid static analysis warnings when the
    # premium module is not present in free distributions
    nodes_xfeat = importlib.import_module(".py.nodes_xfeat", package=__package__)
    for name in (
        "ColormapHotGray",
        
        "HoughLinesAccumulator",
        "HoughLinesPlotter",
        "KLargestPoints",
        "PointToLineParams",
        "IntersectImageWithMask",

    ):
        if hasattr(nodes_xfeat, name):
            AddMapping(getattr(nodes_xfeat, name))
except ImportError as e:
    log("Premium nodes ignored", msg_color="BRIGHT_GREEN")