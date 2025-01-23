from .py.log import log
# import importlib.util # for auto import

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def AddMapping(node_class, display_name):
    NODE_CLASS_MAPPINGS[node_class.NAME] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[node_class.NAME] = display_name

# Load Image Nodes
try:
    from .py.nodes_image import ImageResizeTo, Image2BW
    AddMapping(ImageResizeTo, "Image Resize To 🐦‍🔥")
    AddMapping(Image2BW, "Image To Black and White 🐦‍🔥")
except ImportError:
    log("Failed to load image nodes", msg_color="BRIGHT_GREEN")
