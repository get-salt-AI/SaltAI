from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
import torch

from ... import logger
from SaltAI.modules.convert import pil2tensor, tensor2pil, pil2mask
from SaltAI.modules.masking import MaskFilters
from SaltAI import NAME

class SaltCropImageLocation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "top": ("INT", {"default":0, "max": 10000000, "min":0, "step":1}),
                "left": ("INT", {"default":0, "max": 10000000, "min":0, "step":1}),
                "right": ("INT", {"default":256, "max": 10000000, "min":0, "step":1}),
                "bottom": ("INT", {"default":256, "max": 10000000, "min":0, "step":1}),
            },
            "optional": {
                "crop_data_batch": ("CROP_DATA_BATCH",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CROP_DATA_BATCH")
    RETURN_NAMES = ("images", "crop_data_batch")

    FUNCTION = "crop"
    CATEGORY = f"{NAME}/Image/Process"

    def crop(self, images, top=0, left=0, right=256, bottom=256, crop_data_batch=None):
        cropped_images = []
        crop_data_list = []
        master_size = None
        
        for i, image in enumerate(images):
            image = tensor2pil(image)
            img_width, img_height = image.size

            if crop_data_batch is not None:
                if len(crop_data_batch) >= i:
                    crop_size, (crop_left, crop_top, crop_right, crop_bottom) = crop_data_batch[i]
                else:
                    cropped_images.append(pil2tensor(image))
                    crop_data_list.append((image.size, (0, 0, image.size[0], image.size[1])))
                    continue
            else:
                crop_top = max(top, 0)
                crop_left = max(left, 0)
                crop_bottom = min(bottom, img_height)
                crop_right = min(right, img_width)

            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top

            if crop_width <= 0 or crop_height <= 0:
                errmsg = "Invalid crop dimensions."
                logger.error(errmsg)
                raise ValueError(errmsg)
            
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            crop_data = (crop.size, (crop_left, crop_top, crop_right, crop_bottom))
            if not master_size:
                master_size = (((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8))
            crop = crop.resize(master_size)
            
            cropped_images.append(pil2tensor(crop))
            crop_data_list.append(crop_data)
        
        cropped_images_batch = torch.cat(cropped_images, dim=0)
        return (cropped_images_batch, crop_data_list)
    

class SaltImagePasteCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "crop_images": ("IMAGE",),
                    "crop_data_batch": ("CROP_DATA_BATCH",),
                    "crop_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "crop_sharpening": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                }
            }
            
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks")

    FUNCTION = "image_paste_crop"
    CATEGORY = f"{NAME}/Image/Process"

    def image_paste_crop(self, images, crop_images, crop_data_batch=None, crop_blending=0.25, crop_sharpening=0):
        pasted_images = []
        pasted_masks = []

        for i, image in enumerate(images):
            image = tensor2pil(image)
            crop_image = tensor2pil(crop_images[i].unsqueeze(0))
            
            if crop_data_batch is not None:
                crop_data = crop_data_batch[i]
                pasted_image, pasted_mask = self.paste_image(image, crop_image, crop_data, crop_blending, crop_sharpening)
            else:
                errmsg = f"No valid crop data found for Image {i}"
                logger.warning(errmsg)
                pasted_image = pil2tensor(image)
                pasted_mask = pil2tensor(Image.new("RGB", image.size, (0,0,0)))
            
            pasted_images.append(pasted_image)
            pasted_masks.append(pasted_mask)

        pasted_images_batch = torch.cat(pasted_images, dim=0)
        pasted_masks_batch = torch.cat(pasted_masks, dim=0)

        return (pasted_images_batch, pasted_masks_batch)

    def paste_image(self, image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1):
    
        def lingrad(size, direction, white_ratio):
            image = Image.new('RGB', size)
            draw = ImageDraw.Draw(image)
            if direction == 'vertical':
                black_end = int(size[1] * (1 - white_ratio))
                range_start = 0
                range_end = size[1]
                range_step = 1
                for y in range(range_start, range_end, range_step):
                    if y <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_value = int(((y - black_end) / (size[1] - black_end)) * 255)
                        color = (color_value, color_value, color_value)
                    draw.line([(0, y), (size[0], y)], fill=color)
            elif direction == 'horizontal':
                black_end = int(size[0] * (1 - white_ratio))
                range_start = 0
                range_end = size[0]
                range_step = 1
                for x in range(range_start, range_end, range_step):
                    if x <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_value = int(((x - black_end) / (size[0] - black_end)) * 255)
                        color = (color_value, color_value, color_value)
                    draw.line([(x, 0), (x, size[1])], fill=color)
                    
            return image.convert("L")
    
        crop_size, (left, top, right, bottom) = crop_data
        crop_image = crop_image.resize(crop_size)
        
        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image = crop_image.filter(ImageFilter.SHARPEN)

        blended_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
        blended_mask = Image.new('L', image.size, 0)
        crop_padded = Image.new('RGBA', image.size, (0, 0, 0, 0))
        blended_image.paste(image, (0, 0))
        crop_padded.paste(crop_image, (left, top))
        crop_mask = Image.new('L', crop_image.size, 0)
        
        if top > 0:
            gradient_image = ImageOps.flip(lingrad(crop_image.size, 'vertical', blend_amount))
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if left > 0:
            gradient_image = ImageOps.mirror(lingrad(crop_image.size, 'horizontal', blend_amount))
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if right < image.width:
            gradient_image = lingrad(crop_image.size, 'horizontal', blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if bottom < image.height:
            gradient_image = lingrad(crop_image.size, 'vertical', blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        crop_mask = ImageOps.invert(crop_mask)
        blended_mask.paste(crop_mask, (left, top))
        blended_mask = blended_mask.convert("L")
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (pil2tensor(blended_image.convert("RGB")), pil2tensor(blended_mask.convert("RGB")))
    

class SaltMaskCropRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "padding": ("INT",{"default": 24, "min": 0, "max": 4096, "step": 1}),
                "region_type": (["dominant", "minority"],),
            }
        }
    
    RETURN_TYPES = ("MASK", "CROP_DATA_BATCH", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_masks", "crop_data_batch", "top_int", "left_int", "right_int", "bottom_int", "width_int", "height_int")
    
    FUNCTION = "mask_crop_region"
    CATEGORY = f"{NAME}/Masking/Process"
    
    def mask_crop_region(self, masks, padding=24, region_type="dominant"):
        N = len(masks)
        cropped_masks = []
        crop_data_list = []
        master_size = None
        
        for n in range(N):
            mask = masks[n]
            mask_pil = tensor2pil(mask)
            if not master_size:
                master_size = mask_pil.size
            region_mask, crop_data = MaskFilters.crop_region(mask_pil, region_type, padding)
            region_mask = region_mask.resize(master_size)
            region_tensor = pil2mask(region_mask)
            cropped_masks.append(region_tensor)
            crop_data_list.append(crop_data)

        cropped_masks_batch = torch.cat(cropped_masks, dim=0)

        # Extract crop data
        top_int, left_int, right_int, bottom_int = crop_data_list[0][1]
        width_int, height_int = cropped_masks_batch.shape[2], cropped_masks_batch.shape[1]

        return (cropped_masks_batch, crop_data_list, top_int, left_int, right_int, bottom_int, width_int, height_int)

    
class SaltBatchCropDataExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data_batch": ("CROP_DATA_BATCH",),
                "index": ("INT", {"min": 0}),
            }
        }

    RETURN_TYPES = ("CROP_DATA_BATCH", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("crop_data_batch", "width", "height", "top", "left", "right", "bottom")

    FUNCTION = "extract"
    CATEGORY = f"{NAME}/Masking/Process"

    def extract(self, crop_data_batch, index):
        if index < 0 or index >= len(crop_data_batch):
            errmsg = "Index out of range"
            logger.error(errmsg)
            raise ValueError(errmsg)

        try:
            crop_size, (crop_left, crop_top, crop_right, crop_bottom) = crop_data_batch[index]
            width = crop_right - crop_left
            height = crop_bottom - crop_top
            single_crop_data = [(crop_size, (crop_left, crop_top, crop_right, crop_bottom))]
            return single_crop_data, width, height, crop_top, crop_left, crop_right, crop_bottom
        except Exception as e:
            logger.warning(e)
            return [((0, 0), (0, 0, 0, 0))], 0, 0, 0, 0, 0, 0
    
NODE_CLASS_MAPPINGS = {
    "SaltCropImageLocation": SaltCropImageLocation,
    "SaltImagePasteCrop": SaltImagePasteCrop,
    "SaltMaskCropRegion": SaltMaskCropRegion,
    "SaltBatchCropDataExtractor": SaltBatchCropDataExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltCropImageLocation": "Crop Batch Image Location",
    "SaltImagePasteCrop": "Crop Batch Image Paste",
    "SaltMaskCropRegion": "Mask Batch Crop Region",
    "SaltBatchCropDataExtractor": "Extract Crop Data from Batch"
}