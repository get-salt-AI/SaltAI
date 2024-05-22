import torch

from SaltAI.modules.convert import mask2pil, pil2tensor
from SaltAI import NAME

class SaltMasksToImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    CATEGORY = f"{NAME}/Masking"
    FUNCTION = "convert"

    def convert(self, masks):
        images = []
        for mask in masks:
            images.append(pil2tensor(mask2pil(mask)))
        images = torch.cat(images, dim=0)
        return (images, )
    
NODE_CLASS_MAPPINGS = {
    "SaltMasksToImages": SaltMasksToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltMasksToImages": "Masks to Tensors",
}