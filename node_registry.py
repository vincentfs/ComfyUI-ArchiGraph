#--------------------------------------------------------------------------------#
# Global list to hold registered nodes
#--------------------------------------------------------------------------------#
REGISTERED_NODES = []

def comfy_node(cls):
    """ComfyUI decorator"""
    REGISTERED_NODES.append(cls)
    return cls