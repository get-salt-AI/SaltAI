import torch

from ... import logger
from PIL import Image, ImageOps

from SaltAI.modules.convert import cv2pil, pil2cv, tensor2pil, pil2tensor, mask2pil, masks2pils

load_color_transfer_nodes = True
try:
    from python_color_transfer.color_transfer import ColorTransfer
except ImportError:
    errmsg = "\nUnable to import `python_color_transfer.color_transfer`. Is `python-color-transfer` pip package installed? Skipping python color transfer nodes.\n"
    logger.warning(errmsg)
    load_color_transfer_nodes = False

load_image_blending_modes = True
try:
    import pilgram
except ImportError:
    errmsg = "\nUnable to load Pilgram module. Is it installed? Skipping `PlaiLabsImageBlendingModes` node.\n"
    logger.warning(errmsg)
    load_image_blending_modes = False
    pass

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
                errmsg = f"Invalid mode `{mode}` selected for {self.__class__.__name__}"
                logger.warning(errmsg)
                res = target_image

            results.append(res)

        results = torch.cat(results, dim=0)

        return (results, )
    

class SaltRGBAFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"min": 0, "max": 1.0, "step": 0.01, "default": 0.5}),
                "invert_mask": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)

    FUNCTION = "composite"
    CATEGORY = "SALT/Image/Composite"

    def composite(self, image, mask, threshold, invert_mask):
        img = tensor2pil(image)
        msk = mask2pil(mask)

        msk = msk.convert("L")
        img = img.convert("RGBA")

        img_ratio = img.size[0] / img.size[1]
        msk_ratio = msk.size[0] / msk.size[1]

        if img_ratio > msk_ratio:
            scale_factor = img.size[1] / msk.size[1]
            new_size = (int(msk.size[0] * scale_factor), img.size[1])
        else:
            scale_factor = img.size[0] / msk.size[0]
            new_size = (img.size[0], int(msk.size[1] * scale_factor))

        msk = msk.resize(new_size, Image.Resampling.BILINEAR)

        pad_mask = Image.new("L", img.size, 0)

        x = (img.size[0] - msk.size[0]) // 2
        y = (img.size[1] - msk.size[1]) // 2
        pad_mask.paste(msk, (x, y))

        thresh = int(threshold * 255)
        pad_mask = pad_mask.point(lambda p: 255 if p > thresh else 0)

        if invert_mask:
            pad_mask = ImageOps.invert(pad_mask)

        rgba_image = Image.new("RGBA", img.size, (0, 0, 0, 0))
        rgba_image.paste(img, (0, 0), pad_mask)

        return (pil2tensor(rgba_image),)


class SaltImageBlendingModes:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "mode": ([
                    "normal",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light",
                ],),
                "blend_percentage": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "blend"
    CATEGORY = "SALT/Image/Composite"

    def blend(self, images_a, images_b, mode, blend_percentage, masks=None):
        blended_images = []

        if not isinstance(blend_percentage, list):
            blend_percentage = [blend_percentage]
        
        if isinstance(masks, torch.Tensor):
            masks = masks2pils(masks)

        for i in range(len(images_a)):
            img_a = tensor2pil(images_a[i].unsqueeze(0))
            img_b = tensor2pil(images_b[i if i < len(images_b) else -1].unsqueeze(0))
            img_b_resized = img_b.resize(img_a.size, Image.Resampling.BILINEAR).convert(img_a.mode)

            out_image = getattr(pilgram.css.blending, mode)(img_a, img_b_resized)

            if masks:
                mask_resized = masks[i if i < len(masks) else -1].resize(img_a.size, Image.Resampling.BILINEAR).convert('L')
                black_image = Image.new("L", img_a.size, 0)  # Ensure this black image matches the size
                blend_mask = Image.blend(black_image, mask_resized, blend_percentage[i if i < len(blend_percentage) else -1])
                final_image = Image.composite(out_image, img_a, blend_mask)
            else:
                blend_intensity = int(255 * blend_percentage[i if i < len(blend_percentage) else -1])
                blend_mask = Image.new("L", img_a.size, blend_intensity)
                final_image = Image.composite(out_image, img_a, blend_mask)

            blended_images.append(pil2tensor(final_image))

        blended_images_batch = torch.cat(blended_images, dim=0)
        return (blended_images_batch,)
    

NODE_CLASS_MAPPINGS = {
    "SaltRGBAFromMask": SaltRGBAFromMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltRGBAFromMask": "Mask with Alpha (Transparency)"
}

if load_color_transfer_nodes:
    NODE_CLASS_MAPPINGS["SAIColorTransfer"] = SAIColorTransfer
    NODE_DISPLAY_NAME_MAPPINGS["SAIColorTransfer"] = "Color Transfer"

if load_image_blending_modes:
    NODE_CLASS_MAPPINGS["SaltImageBlendingModes"] = SaltImageBlendingModes
    NODE_DISPLAY_NAME_MAPPINGS["SaltImageBlendingModes"] = "Composite Images"