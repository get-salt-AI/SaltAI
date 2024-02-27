import os

import folder_paths

from salt_comfyui.modules.node_importer import ModuleLoader

ROOT = os.path.abspath(os.path.dirname(__file__))
NAME = "Salt.AI"
PACKAGE = "salt_comfyui"
NODES_DIR = os.path.join(ROOT, 'nodes')
EXTENSION_WEB_DIRS = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Load modules
module_timings = {}
module_loader = ModuleLoader(PACKAGE)
module_loader.load_modules(NODES_DIR)

# Mappings
# EXTENSION_WEB_DIRS = module_loader.EXTENSION_WEB_DIRS # Whenever this is a thing
NODE_CLASS_MAPPINGS = module_loader.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = module_loader.NODE_DISPLAY_NAME_MAPPINGS

# Timings and such
module_loader.report(NAME)

# Add .results format (for future, in case PlaiLabs isn't refactored for public)
folder_paths.supported_pt_extensions.add('.results')
if '.results' in folder_paths.supported_pt_extensions:
    print("\nAdded LoRa extension format: .results")
else:
    print("\nUnable to add LoRa extension format: .results")

# Export nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
