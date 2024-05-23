import os
import torch

from folder_paths import models_dir

from SaltAI.modules.convert import pil2tensor, tensor2pil, image2mask
from SaltAI import NAME

class SaltCLIPSegLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "CIDAS/clipseg-rd64-refined", "multiline": False}),
            },
        }

    RETURN_TYPES = ("CLIPSEG_MODEL",)
    RETURN_NAMES = ("clipseg_model",)
    FUNCTION = "clipseg_model"

    CATEGORY = f"{NAME}/Loaders"

    def clipseg_model(self, model):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        cache = os.path.join(models_dir, 'clipseg')

        inputs = CLIPSegProcessor.from_pretrained(model, cache_dir=cache)
        model = CLIPSegForImageSegmentation.from_pretrained(model, cache_dir=cache)

        return ( (inputs, model), ) 
            
# CLIPSeg Node
        
class SaltCLIPSegMasking:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default":"", "multiline": False}),
            },
            "optional": {
                "clipseg_model": ("CLIPSEG_MODEL",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "mask_images")
    FUNCTION = "CLIPSeg_image"

    CATEGORY = f"{NAME}/Masking"

    def CLIPSeg_image(self, images, text=None, clipseg_model=None):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        masks = []
        image_masks = []
        master_size = None
        
        for image in images:

            image = tensor2pil(image)
            cache = os.path.join(models_dir, 'clipseg')

            if not master_size:
                master_size = image.size

            if clipseg_model:
                inputs = clipseg_model[0]
                model = clipseg_model[1]
            else:
                inputs = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache)
                model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache)

            with torch.no_grad():
                result = model(**inputs(text=text, images=image, padding=True, return_tensors="pt"))

            tensor = torch.sigmoid(result[0])
            mask = (tensor - tensor.min()) / tensor.max()
            mask = mask.unsqueeze(0)
            mask = tensor2pil(mask)
            mask = mask.resize(master_size)
            mask_image_tensor = pil2tensor(mask.convert("RGB"))
            mask_tensor = image2mask(mask_image_tensor)
            masks.append(mask_tensor)
            image_masks.append(mask_image_tensor)

        masks = torch.cat(masks, dim=0)
        image_masks = torch.cat(image_masks, dim=0)
                
        return (masks, image_masks)
    
NODE_CLASS_MAPPINGS = {
    "SaltCLIPSegLoader": SaltCLIPSegLoader,
    "SaltCLIPSegMasking": SaltCLIPSegMasking
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltCLIPSegLoader": "CLIPSeg Model Loader (Salt)",
    "SaltCLIPSegMasking": "Batch Image CLIPSeg Masking"
}