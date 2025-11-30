from .py.log import log
# import importlib.util # for auto import

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def AddMapping(node_class):
    # Add AG prefix to original name for avoid conflict
    NODE_CLASS_MAPPINGS["AG "+node_class.NAME] = node_class
    NODE_DISPLAY_NAME_MAPPINGS["AG "+node_class.NAME] = node_class.NAME+" 🐦‍🔥"

#--------------------------------------------------------------------------------#
# Load Image Nodes
#--------------------------------------------------------------------------------#
try:
    from .py.nodes_utils import JsonSave, JsonLoad, NparraySave, NparrayLoad, \
        Printf, SliceList, SliceSublist, ImageResizeTo, Image2BW
    AddMapping(JsonSave)
    AddMapping(JsonLoad)
    AddMapping(NparraySave)
    AddMapping(NparrayLoad)
    
    AddMapping(Printf)
    AddMapping(SliceList)
    AddMapping(SliceSublist)
    AddMapping(ImageResizeTo)
    AddMapping(Image2BW)
except ImportError as e:
    log("Failed to load util nodes", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")

#--------------------------------------------------------------------------------#
# Load OpenCV Nodes
#--------------------------------------------------------------------------------#
try:
    from .py.nodes_opencv import Image2Nparray, Nparray2Image, \
        Bitwise_not, Threshold, GaussianBlur, MedianBlur, Blur, \
        Canny, HoughLines, HoughLinesP , FindContours, DrawContours, Circles
    AddMapping(Image2Nparray)
    AddMapping(Nparray2Image)
    AddMapping(Printf)
    AddMapping(SliceList)
    AddMapping(SliceSublist)

    AddMapping(Bitwise_not)
    AddMapping(Threshold)
    AddMapping(GaussianBlur)
    AddMapping(MedianBlur)
    AddMapping(Blur)

    AddMapping(Canny)
    AddMapping(HoughLines)
    AddMapping(HoughLinesP)
    AddMapping(FindContours)
    AddMapping(DrawContours)
    AddMapping(Circles)
except ImportError as e:
    log("Failed to load opencv nodes", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")

#--------------------------------------------------------------------------------#
# Load XFeat Nodes
#--------------------------------------------------------------------------------#
try:
    from .py.nodes_xfeat import HoughLinesAccumulator, HoughLinesPlotter, KLargestPoints, PointToLineParams, \
        ColormapHotGray
    AddMapping(HoughLinesAccumulator)
    AddMapping(HoughLinesPlotter)
    AddMapping(KLargestPoints)
    AddMapping(PointToLineParams)

    AddMapping(ColormapHotGray)
except ImportError as e:
    log("Failed to load cv nodes", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")
