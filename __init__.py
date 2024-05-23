import os

import folder_paths

from .modules.node_importer import ModuleLoader
from .modules.log import create_logger
logger = create_logger()


ROOT = os.path.abspath(os.path.dirname(__file__))
NAME = "SALT"
DISPLAY_NAME = "Salt AI"
PACKAGE = "SaltAI"
NODES_DIR = os.path.join(ROOT, 'nodes')
EXTENSION_WEB_DIRS = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"


# Load modules
module_timings = {}
module_loader = ModuleLoader(PACKAGE)
module_loader.load_modules(NODES_DIR)

# Mappings
# EXTENSION_WEB_DIRS = module_loader.EXTENSION_WEB_DIRS # Whenever this is a thing
NODE_CLASS_MAPPINGS = module_loader.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = module_loader.NODE_DISPLAY_NAME_MAPPINGS

# Timings and such
module_loader.report(DISPLAY_NAME)

# Add .results format
folder_paths.supported_pt_extensions.add('.results')
if '.results' in folder_paths.supported_pt_extensions:
    logger.info("\nAdded LoRa extension format: .results")
else:
    logger.warning("\nUnable to add LoRa extension format: .results")

# Export nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
