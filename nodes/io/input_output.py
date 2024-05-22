import io
import inspect
import os
import torch
from PIL import Image
import uuid

from ... import logger
import folder_paths

from pydub import AudioSegment

from SaltAI.modules.convert import tensor2pil, pil2tensor, pil2mask
from SaltAI.modules.types import WILDCARD
from SaltAI.modules.sanitize import sanitize_filename, bool_str

from SaltAI.modules.animation.image_animator import ImageAnimator

def get_relative_path(full_path):
    """Return the relative path to a salt temp directory, or base input directory"""
    parent_dir = os.path.basename(os.path.dirname(full_path))
    if parent_dir == "input" or parent_dir == "temp":
        return os.path.basename(full_path)
    else:
        return os.path.join(parent_dir, os.path.basename(full_path))
    
def log_values(id, input_name, input_type, input_value):
    print(f"[SaltInput_{id}] `{input_name}` ({input_type}) Value:")
    print(input_value)

def is_lambda(v):
    return callable(v) and inspect.isfunction(v) and v.__name__ == "<lambda>"

class SaltInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {}),
                "input_desc": ("STRING", {}),
                "input_type": (["STRING", "FLOAT", "INT", "BOOLEAN", "IMAGE", "MASK", "SEED", "FILE"],),
                "input_value": ("STRING", {"multiline": True, "dynamicPrompts": False})
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_mask": ("MASK",),
                "input_allowed_values": ("STRING", {"default": ""}),
                "user_override_required": ("BOOLEAN", {"default": False}),
                "relative_path": ("BOOLEAN", {"default": False})
            },
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
        input_image=None,
        input_mask=None,
        input_allowed_values="",
        user_override_required=False,
        relative_path=False,
        unique_id=0,
    ):
        src_image = None
        src_file = None
        is_asset = False

        # Is an asset type
        if input_type in ["IMAGE", "MASK", "FILE"]:
            is_asset = True

        # UI Output
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

        out = ""
        if is_asset:
            # Input value must be evaluated first to override input images/masks
            if input_value.strip():
                input_value = input_value.strip()
                # Load image from path from input_value
                if input_value.endswith(('.png', '.jpeg', '.jpg', '.gif', '.webp', '.tiff')):
                    try:
                        src_image = Image.open(input_value).convert("RGBA")
                    except Exception as e:
                        errmsg = f"Error loading image from specified path {input_value}: {e}"
                        logger.warning(errmsg)
                # Passthrough input_value (which should be a path from Salt Backend)
                elif input_type == "FILE":

                    src_file = input_value # if os.path.exists(input_value) else "None"
                    if relative_path:
                        src_file = get_relative_path(src_file)

                    # Log value to console
                    log_values(unique_id, input_name, input_type, src_file)
                    
                    return {"ui": ui, "result": (src_file,)}
                else:
                    errmsg = "Invalid node configuration! Do you mean to use `IMAGE`, `MASK`, or `FILE` input_types?"
                    logger.error(errmsg)
                    raise AttributeError(errmsg)
            elif isinstance(input_image, torch.Tensor):
                # Input `IMAGE` is provided, so we act like a passthrough
                return (input_image, ui)
            elif isinstance(input_mask, torch.Tensor):
                # Input `MASK` is provided, so we act like a passthrough
                return (input_mask, ui)

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
                errmsg = "Unable to determine IMAGE or MASK to load!  Returning image blank"
                logger.warning(errmsg)

                src_blank = Image.new("RGB", (512, 512), (0, 0, 0))
                if input_type == "IMAGE":
                    src_image = pil2tensor(src_blank)
                else:
                    src_image = pil2mask(src_blank)

            # Log value to console
            log_values(unique_id, input_name, input_type, src_image)

            return (src_image, ui)

        # We're still here? We must be dealing with a primitive value
        if input_allowed_values != "" and input_value.strip() not in [o.strip() for o in input_allowed_values.split(',')]:
            errmsg = 'The provided input is not a supported value'
            logger.warning(errmsg)
            raise ValueError(errmsg)


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
        log_values(unique_id, input_name, input_type, out)

        return (out, ui)


class SaltOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {}),
                "output_desc": ("STRING", {}),
                "output_type": (
                    ["PNG", "JPEG", "GIF", "WEBP", "AVI", "MP4", "WEBM", "MP3", "WAV", "STRING"],
                ),
                "output_data": (WILDCARD, {}),
            },
            "optional": {
                "video_audio": ("AUDIO", {}),
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
        video_audio=None,
        animation_fps=8,
        animation_quality="DEFAULT",
        unique_id=0,
        output_subdir=None
    ):
        is_asset = False
        asset_id = str(uuid.uuid4())

        # Determine if valid type
        if not isinstance(output_data, torch.Tensor) and not isinstance(output_data, str) and not isinstance(output_data, bytes) and not is_lambda(output_data):
            errmsg = f"Unsupported output_data supplied `{str(type(output_data).__name__)}`. Please provide `IMAGE` (torch.Tensor), `STRING` (str), or `AUDIO` (bytes) input."
            logger.error(errmsg)
            raise ValueError(errmsg)
        
        # Support VHS audio
        if output_type in ["AVI", "MP4", "WEBM", "MP3", "WAV"]:
            if video_audio is not None and is_lambda(video_audio):
                video_audio = video_audio()
            if is_lambda(output_data):
                output_data = output_data()
        
        if video_audio and not isinstance(video_audio, bytes):
            errmsg = f"Unsupported video_audio supplied `{str(type(output_data).__name__)}. Please provide `AUDIO` (bytes)"
            logger.error(errmsg)
            raise ValueError(errmsg)

        # Is asset? I may have misunderstood this part
        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM", "MP3", "WAV"]:
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
            errmsg = f"Unable to create output directory `{output_path}`"
            logger.warning(errmsg)

        results = []
        if output_type in ("PNG", "JPEG") and isinstance(output_data, torch.Tensor):
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
                        logger.info(f"Saved image to `{image_path}`")
                    else:
                        errmsg = f"Unable to save image to `{image_path}`"
                        logger.warning(errmsg)

            except Exception as e:
                errmsg = f"Unknown exception {e}"
                logger.error(errmsg)
                raise e

        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"] and isinstance(output_data, torch.Tensor):
            # Save animation file
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")
            animator = ImageAnimator(
                output_data, fps=int(animation_fps), quality=animation_quality
            )
            animator.save_animation(filename, format=output_type, audio=video_audio)
            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })
            if os.path.exists(filename):
                logger.info(f"[SALT] Saved file to `{filename}`")
            else:
                errmsg = f"[SALT] Unable to save file to `{filename}`"
                logger.warning(errmsg)
        elif output_type in ["MP3", "WAV"] and isinstance(output_data, bytes):
            # Save audio file
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")

            audio_buffer = io.BytesIO(output_data)
            audio = AudioSegment.from_file(audio_buffer)

            if output_type == "MP3":
                audio.export(filename, format="mp3")
            else:
                audio.export(filename, format="wav")

            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })

            if os.path.exists(filename):
                logger.info(f"Saved file to `{filename}`")
            else:
                errmsg = f"Unable to save file to `{filename}`"
                logger.warning(errmsg)

        else:
            # Assume string output
            if output_type == "STRING":
                results.append(str(output_data))

        # Output Dictionary
        ui = {
            "ui": {
                "salt_metadata": [
                    {
                        "salt_id": unique_id,
                        "salt_reference_uuid": asset_id,
                        "salt_description": output_desc,
                        "salt_asset": is_asset,
                        "salt_file_extension": output_type,
                    }
                ],
                "salt_output": results
            }
        }

        # Assign images for previews of supported types
        if output_type in ["PNG", "GIF", "WEBP", "JPEG"] and results:
            ui["ui"].update({"images": results})

        # Print to log
        logger.info(f"[SaltOutput_{unique_id}] Output:")
        from pprint import pprint

        pprint(ui, indent=4) # Not converting this to logger yet to get the rest done - Daniel

        return ui
        

class SaltInfo:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_title": ("STRING", {}),
                "workflow_description": ("STRING", {}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("title", "description")

    FUNCTION = "info"
    CATEGORY = "SALT/IO"

    def info(self, workflow_title, workflow_description, unique_id=0):

        logger.info(f"[SaltInfo_{unique_id}] Workflow Info:")
        logger.info(f"Title: {workflow_title}")
        logger.info(f"Description: {workflow_description}")

        return (workflow_title, workflow_description)


# Node Export Manifest
NODE_CLASS_MAPPINGS = {
    "SaltInput": SaltInput, 
    "SaltOutput": SaltOutput,
    "SaltInfo": SaltInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltInput": "Salt Workflow Input",
    "SaltOutput": "Salt Workflow Output",
    "SaltInfo": "Salt Workflow Info"
}
