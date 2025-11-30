from .py.log import log
from .node_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./js/v1"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

log('Successfully Initiated 💥🚀')