# CORE NODES

from .nodes_core import NODE_CLASS_MAPPINGS as nodes_core_classes
from .nodes_core import NODE_DISPLAY_NAME_MAPPINGS as nodes_core_display_mappings

# TOOL NODES

from .nodes_tools import NODE_CLASS_MAPPINGS as nodes_tools_classes
from .nodes_tools import NODE_DISPLAY_NAME_MAPPINGS as nodes_tools_display_mappings

# COMPILE NODE MAPPINGS

NODE_CLASS_MAPPINGS = nodes_core_classes
NODE_DISPLAY_NAME_MAPPINGS = nodes_core_display_mappings

NODE_CLASS_MAPPINGS.update(nodes_tools_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(nodes_tools_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
