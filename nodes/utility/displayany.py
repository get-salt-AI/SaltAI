import json
import torch

from ... import logger
from SaltAI.modules.types import WILDCARD

class SaltDisplayAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": (WILDCARD, {}),
            },
            "optional": {
                "double_linebreaks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("output", )
    WEB_DIRECTORY = "./web"

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "SALT/Utility"

    def main(self, input_value=None, double_linebreaks=True):
        if input_value is None:
            return {"ui": {"text":""}, "result": ("",)}

        if isinstance(input_value, (str, int, float, bool)):
            value = str(input_value)
            if double_linebreaks:
                value = value.replace("\n", "\n\n")
        elif isinstance(input_value, dict):
            try:
                value = json.dumps(input_value)
            except Exception:
                value = f"Data type {type(input_value).__name__} could not be serialized."
        elif isinstance(input_value, list):
            try:
                value = json.dumps(input_value)
                if double_linebreaks:
                    value = value.replace("\n", "\n\n")
            except Exception:
                value = f"Data type {type(input_value).__name__} could not be serialized."
        elif isinstance(input_value, torch.Tensor):
            value = str(input_value.shape)
        else:
            value = f"Data type {type(input_value).__name__} cannot be displayed"

        logger.info(value)
        return {"ui": {"text": value}, "result": (input_value,)}


NODE_CLASS_MAPPINGS = {
    "SaltDisplayAny": SaltDisplayAny,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltDisplayAny": "SaltDisplayAny",
}
