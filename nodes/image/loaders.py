import zipfile
from PIL import Image
from io import BytesIO
import torch

from ... import logger
from SaltAI.modules.convert import pil2tensor

class SaltLoadImageZip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {}),
                "resize_images_to_first": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    CATEGORY = "SALT/Image/Loaders"
    FUNCTION = "load_images"

    def load_images(self, path: str, resize_images_to_first: bool = True):
        supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp')
        images = []
        first_image_size = None
        
        with zipfile.ZipFile(path, 'r') as z:
            for file_name in z.namelist():
                if file_name.lower().endswith(supported_formats):
                    with z.open(file_name) as file:
                        image = Image.open(BytesIO(file.read()))
                        if first_image_size is None:
                            first_image_size = image.size
                        if image.size == first_image_size or resize_images_to_first:
                            images.append(image if image.size == first_image_size else self.resize_right(image, first_image_size))

        if not images:
            errmsg = f"The input zip `{path}` does not contain any valid images!"
            logger.error(errmsg)
            raise ValueError(errmsg)

        images = [pil2tensor(img) for img in images]
        images = torch.cat(images, dim=0)

        return (images, )

    def resize_right(self, image, target_size):
        img_ratio = image.width / image.height
        target_ratio = target_size[0] / target_size[1]
        resize_width, resize_height = (
            (target_size[0], round(target_size[0] / img_ratio)) if target_ratio > img_ratio else
            (round(target_size[1] * img_ratio), target_size[1])
        )
        image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        x_crop, y_crop = (resize_width - target_size[0]) // 2, (resize_height - target_size[1]) // 2
        return image.crop((x_crop, y_crop, x_crop + target_size[0], y_crop + target_size[1]))
    
NODE_CLASS_MAPPINGS = {
    "SaltLoadImageZip": SaltLoadImageZip
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltLoadImageZip": "Load Images from ZIP"
}