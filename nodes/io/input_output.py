import os
import numpy as np
import torch
from PIL import Image
import uuid

import folder_paths

from SaltAI import ROOT
from SaltAI.modules.convert import tensor2pil, pil2tensor, pil2mask
from SaltAI.modules.types import WILDCARD
from SaltAI.modules.sanitize import sanitize_filename, bool_str

from SaltAI.modules.animation.image_animator import ImageAnimator

class SaltInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {}),
                "input_desc": ("STRING", {}),
                "input_type": (["STRING", "FLOAT", "INT", "BOOLEAN", "IMAGE", "MASK", "SEED"],),
                "input_value": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "user_override_required": ("BOOLEAN", {}),
            },
            "optional": {"input_image": ("IMAGE",), "input_mask": ("MASK",), "input_allowed_values": ("STRING",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("value",)

    FUNCTION = "input"
    CATEGORY = "SALT/IO"

    def input(
        self,
        input_name,
        input_desc,
        input_value,
        input_type,
        user_override_required,  # only used for upstream input validation
        input_image=None,
        input_mask=None,
        input_allowed_values=None,
        unique_id=0,
    ):
        src_image = None
        is_asset = False

        if input_type in ["IMAGE", "MASK"]:
            is_asset = True

        ui = {
            "ui": {
                "salt_input": [{
                    "id": unique_id,
                    "name": input_name or "input_" + str(unique_id),
                    "description": input_desc or "",
                    "asset": is_asset or False,
                    "type": input_type or "string",
                    "value": input_value or "",
                }]
            }
        }

        if is_asset:
            if isinstance(input_image, torch.Tensor):
                # Input `IMAGE` is provided, so we act like a passthrough
                return (input_image, ui)
            elif isinstance(input_mask, torch.Tensor):
                # Input `MASK` is provided, so we act like a passthrough
                return (input_mask, ui)
            elif input_value.strip():
                # Load image from path from input_value
                try:
                    src_image = Image.open(input_value.strip()).convert("RGBA")
                except Exception as e:
                    print(f"Error loading image from specified path {input_value}: {e}")

            if src_image:
                if input_type == "MASK":
                    # If it's a mask and the image has an alpha channel, extract it
                    if src_image.mode == "RGBA":
                        alpha_channel = src_image.split()[-1]
                        src_image = pil2mask(alpha_channel.convert("L"))
                    # If no alpha channel, convert the whole image to grayscale as a mask (could be bitwise representation)
                    else:
                        src_image = pil2mask(src_image.convert("L"))
                elif input_type == "IMAGE":
                    # Ensure image is in RGB for `IMAGE`` data type
                    src_image = pil2tensor(src_image.convert("RGB"))

            else:
                # Gracefully allow execution to continue, provided a black image (to hopefully signal issue?)
                print("[WARNING] Unable to determine IMAGE or MASK to load!")
                print("[WARNING] Returning image blank")
                src_blank = Image.new("RGB", (512, 512), (0, 0, 0))
                if input_type == "IMAGE":
                    src_image = pil2tensor(src_blank)
                else:
                    src_image = pil2mask(src_blank)

            return (src_image, ui)

        # We're still here? We must be dealing with a primitive value
        if input_allowed_values is not None and input_value.strip() not in [o.strip() for o in input_allowed_values.split(',')]:
            raise ValueError('The provided input is not a supported value')


        out = ""
        match input_type:
            case "STRING":
                out = str(input_value)
            case "INT":
                out = int(input_value)
            case "SEED":
                out = int(input_value)
            case "FLOAT":
                out = float(input_value)
            case "BOOLEAN":
                out = bool_str(input_value)
            case _:
                out = input_value

        # Log value to console
        print(f"[SaltInput_{unique_id}] `{input_name}` ({input_type}) Value:")
        print(out)

        return (out, ui)


class SaltOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {}),
                "output_desc": ("STRING", {}),
                "output_type": (
                    ["PNG", "JPEG", "GIF", "WEBP", "AVI", "MP4", "WEBM", "STRING"],
                ),
                "output_data": (WILDCARD, {}),
            },
            "optional": {
                "animation_fps": ("INT", {"min": 1, "max": 60, "default": 8}),
                "animation_quality": (["DEFAULT", "HIGH"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "output_subdir": None},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    FUNCTION = "output"
    CATEGORY = "SALT/IO"

    def output(
        self,
        output_name,
        output_desc,
        output_type,
        output_data,
        animation_fps=8,
        animation_quality="DEFAULT",
        unique_id=0,
        output_subdir=None
    ):
        is_asset = False
        asset_id = str(uuid.uuid4())

        # Determine if valid type
        if output_type.strip() == "" or output_type not in [
            "GIF",
            "WEBP",
            "AVI",
            "MP4",
            "WEBM",
        ]:
            if isinstance(output_data, torch.Tensor):
                output_type = "JPEG" if output_type == "JPEG" else "PNG"
            elif isinstance(output_data, str):
                output_type = "STRING"
            else:
                raise ValueError(
                    "Unsupported `output_type` supplied. Please provide `IMAGE` or `STRING` input."
                )

        # Is asset? I may have misunderstood this part
        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"]:
            is_asset = True

        # Determine output name, and sanitize if input (for filesystem)
        if output_name.strip() == "":
            output_name = "output_" + str(unique_id)
        else:
            output_name = sanitize_filename(output_name)

        # Create output dir based on uuid4
        subfolder = os.path.join(output_subdir or '', asset_id)
        output_path = os.path.join(folder_paths.get_output_directory(), subfolder)

        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            print(f"[SALT] Unable to create output directory `{output_path}`")

        results = []
        if output_type in ("PNG", "JPEG"):
            # Save all images in the tensor batch as specified by output_type
            try:
                for index, img in enumerate(output_data):
                    pil_image = tensor2pil(img)
                    file_prefix = output_name.strip().replace(" ", "_")
                    file_ext = f".{output_type.lower()}"
                    filename = f"{file_prefix}_{index:04d}{file_ext}"
                    image_path = os.path.join(output_path, filename)
                    pil_image.save(image_path, output_type)
                    results.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    if os.path.exists(image_path):
                        print(f"[SALT] Saved image to `{image_path}`")
                    else:
                        print(f"[SALT] Unable to save image to `{image_path}`")

            except Exception as e:
                raise e

        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"]:
            # Save animation file
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")
            animator = ImageAnimator(
                output_data, fps=int(animation_fps), quality=animation_quality
            )
            animator.save_animation(filename, format=output_type)
            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })
            if os.path.exists(filename):
                print(f"[SALT] Saved file to `{filename}`")
            else:
                print(f"[SALT] Unable to save file to `{filename}`")
        else:
            # Prepare output string
            if output_type == "STRING":
                results.append(str(output_data))

        # Output Dictionary
        ui = {
            "ui": {
                "salt_id": unique_id,
                "salt_reference_uuid": asset_id,
                "salt_description": output_desc,
                "salt_asset": is_asset,
                "salt_file_extension": output_type,
                "salt_output": results
            }
        }

        # Assign images for previews of supported types
        if output_type in ["PNG", "GIF", "WEBP", "JPEG"] and results:
            ui["ui"].update({"images": results})

        # Print to log
        print(f"[SaltOutput_{unique_id}] Output:")
        from pprint import pprint

        pprint(ui, indent=4)

        return ui


# Node Export Manifest
NODE_CLASS_MAPPINGS = {"SaltInput": SaltInput, "SaltOutput": SaltOutput}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltInput": "Salt Flow Input",
    "SaltOutput": "Salt Flow Output",
}
