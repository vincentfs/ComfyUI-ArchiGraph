from .py.log import log
import importlib
from .node_registry import REGISTERED_NODES
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

node_modules = [
    ".py.nodes_photoedit",
    ".py.nodes_utils",
    ".py.nodes_opencv",
    ".py.nodes_xfeat",
    # New categories can be added here
]

# import modules
try:
    for module_path in node_modules:
        importlib.import_module(module_path, package=__package__)
except ImportError as e:
    log(f"Failed to load nodes from {module_path}", msg_color="BRIGHT_GREEN")
    log(e, msg_color="BRIGHT_GREEN")

# register all nodes
for node_class in REGISTERED_NODES:
    try:
        AddMapping(node_class)
    except Exception as e:
        if hasattr(node_class, "NAME"):
            log(f"Failed to register nodes {node_class.NAME}", msg_color="BRIGHT_GREEN")
        else:
            log("Failed to register nodes", msg_color="BRIGHT_GREEN")
        log(e, msg_color="BRIGHT_GREEN")