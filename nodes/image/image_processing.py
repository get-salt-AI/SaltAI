import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

load_color_transfer_nodes = True
try:
    from python_color_transfer.color_transfer import ColorTransfer
except ImportError:
    print("\nUnable to import `python_color_transfer.color_transfer`. Is `python-color-transfer` pip package installed? Skipping python color transfer nodes.\n")
    load_color_transfer_nodes = False

load_fa_nodes = True

from SaltAI.modules.convert import cv2pil, pil2cv, tensor2pil, pil2tensor, mask2pil

class SAIColorTransfer:
    def __init__(self):
        self.ct = ColorTransfer()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_images": ("IMAGE",),
                "source_images": ("IMAGE",),
                "mode": (["pdf_regrain", "mean_transfer", "lab_transfer"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "transfer"
    CATEGORY = "SALT/Image/Process"

    def transfer(self, target_images, source_images, mode):

        if target_images.shape[0] != source_images.shape[0]:
            repeat_factor = target_images.shape[0] // source_images.shape[0]
            source_images = source_images.repeat(repeat_factor, 1, 1, 1)

        results = []
        for target_image, source_image in zip(target_images, source_images):

            target_pil = tensor2pil(target_image)
            source_pil = tensor2pil(source_image)
            source_pil = source_pil.resize(target_pil.size)

            if mode == "pdf_regrain":
                res = pil2tensor(cv2pil(self.ct.pdf_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil), regrain=True)))
            elif mode == "mean_transfer":
                res = pil2tensor(cv2pil(self.ct.mean_std_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil))))
            elif mode == "lab_transfer":
                res = pil2tensor(cv2pil(self.ct.lab_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil))))
            else:
                print(f"Invalid mode `{mode}` selected for {self.__class__.__name__}")
                res = target_image

            results.append(res)

        results = torch.cat(results, dim=0)

        return (results, )


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if load_color_transfer_nodes:
    NODE_CLASS_MAPPINGS["SAIColorTransfer"] = SAIColorTransfer
    NODE_DISPLAY_NAME_MAPPINGS["SAIColorTransfer"] = "Color Transfer"
