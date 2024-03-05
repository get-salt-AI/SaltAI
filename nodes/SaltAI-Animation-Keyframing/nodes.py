import random
import time
import torch

from pprint import pprint
from tqdm import tqdm

import nodes
from comfy.utils import ProgressBar

from .modules.transform import (
    PerlinNoise, 
    OrganicPerlinCameraScheduler, 
    generate_frame, 
    movement_modes, 
    easing_functions, 
    edge_modes,
    zoom_presets,
    horizontal_pan_presets,
    vertical_pan_presets
)
from .modules.utils import pil2mask, pil2tensor, tensor2pil

def log_curve(label, value):
    print(f"\t\033[1m\033[93m{label}:\033[0m")
    pprint(value, indent=4)

class OPAC:
    """
        Generates semi-random keyframes for zoom, spin, and translation based on specified start and end ranges,
        with individual tremor scale controls for each parameter, allowing for organic variation using Perlin noise.
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_wiggle": ("BOOLEAN", {"default": True}),
                "frame_count": ("INT", {"default": 48, "min": 1, "max": 500}),
                "zoom_range": ("STRING", {"default": "0.95,1.05"}),
                "zoom_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "angle_range": ("STRING", {"default": "-5,5"}),
                "angle_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trx_range": ("STRING", {"default": "-10,10"}),
                "trx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "try_range": ("STRING", {"default": "-10,10"}),
                "try_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trz_range": ("STRING", {"default": "-10,10"}),
                "trz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotx_range": ("STRING", {"default": "-5,5"}),
                "rotx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "roty_range": ("STRING", {"default": "-5,5"}),
                "roty_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotz_range": ("STRING", {"default": "-5,5"}),
                "rotz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
            },
            "optional": {
                "opac_perlin_settings": ("DICT", {})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom", "angle", "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z")
    FUNCTION = "execute"
    CATEGORY = "SALT/Animation/Keyframing"

    def process_kwargs(self, **kwargs):
        self.use_wiggle = kwargs.get('use_wiggle', True)
        self.frame_count = kwargs.get('frame_count', 48)
        self.zoom_range = tuple(map(float, kwargs.get('zoom_range', "0.95,1.05").split(',')))
        self.zoom_tremor_scale = kwargs.get('zoom_tremor_scale', 0.05)
        self.angle_range = tuple(map(float, kwargs.get('angle_range', "-5,5").split(',')))
        self.angle_tremor_scale = kwargs.get('angle_tremor_scale', 0.05)
        self.trx_range = tuple(map(float, kwargs.get('trx_range', "-10,10").split(',')))
        self.trx_tremor_scale = kwargs.get('trx_tremor_scale', 0.05)
        self.try_range = tuple(map(float, kwargs.get('try_range', "-10,10").split(',')))
        self.try_tremor_scale = kwargs.get('try_tremor_scale', 0.05)
        self.trz_range = tuple(map(float, kwargs.get('trz_range', "-10,10").split(',')))
        self.trz_tremor_scale = kwargs.get('trz_tremor_scale', 0.05)
        self.rotx_range = tuple(map(float, kwargs.get('rotx_range', "-5,5").split(',')))
        self.rotx_tremor_scale = kwargs.get('rotx_tremor_scale', 0.05)
        self.roty_range = tuple(map(float, kwargs.get('roty_range', "-5,5").split(',')))
        self.roty_tremor_scale = kwargs.get('roty_tremor_scale', 0.05)
        self.rotz_range = tuple(map(float, kwargs.get('rotz_range', "-5,5").split(',')))
        self.rotz_tremor_scale = kwargs.get('rotz_tremor_scale', 0.05)

        # Zoom Perlin settings
        self.zoom_octaves = kwargs.get('zoom_octaves', 1)
        self.zoom_persistence = kwargs.get('zoom_persistence', 0.5)
        self.zoom_lacunarity = kwargs.get('zoom_lacunarity', 2.0)
        self.zoom_repeat = kwargs.get('zoom_repeat', 1024)
            
        # Angle Perlin settings
        self.angle_octaves = kwargs.get('angle_octaves', 1)
        self.angle_persistence = kwargs.get('angle_persistence', 0.5)
        self.angle_lacunarity = kwargs.get('angle_lacunarity', 2.0)
        self.angle_repeat = kwargs.get('angle_repeat', 1024)
            
        # Translation Perlin settings (trx, try, trz)
        self.trx_octaves = kwargs.get('trx_octaves', 1)
        self.trx_persistence = kwargs.get('trx_persistence', 0.5)
        self.trx_lacunarity = kwargs.get('trx_lacunarity', 2.0)
        self.trx_repeat = kwargs.get('trx_repeat', 1024)
            
        self.try_octaves = kwargs.get('try_octaves', 1)
        self.try_persistence = kwargs.get('try_persistence', 0.5)
        self.try_lacunarity = kwargs.get('try_lacunarity', 2.0)
        self.try_repeat = kwargs.get('try_repeat', 1024)
            
        self.trz_octaves = kwargs.get('trz_octaves', 1)
        self.trz_persistence = kwargs.get('trz_persistence', 0.5)
        self.trz_lacunarity = kwargs.get('trz_lacunarity', 2.0)
        self.trz_repeat = kwargs.get('trz_repeat', 1024)
            
        # Rotation Perlin settings (rotx, roty, rotz)
        self.rotx_octaves = kwargs.get('rotx_octaves', 1)
        self.rotx_persistence = kwargs.get('rotx_persistence', 0.5)
        self.rotx_lacunarity = kwargs.get('rotx_lacunarity', 2.0)
        self.rotx_repeat = kwargs.get('rotx_repeat', 1024)
        
        self.roty_octaves = kwargs.get('roty_octaves', 1)
        self.roty_persistence = kwargs.get('roty_persistence', 0.5)
        self.roty_lacunarity = kwargs.get('roty_lacunarity', 2.0)
        self.roty_repeat = kwargs.get('roty_repeat', 1024)
            
        self.rotz_octaves = kwargs.get('rotz_octaves', 1)
        self.rotz_persistence = kwargs.get('rotz_persistence', 0.5)
        self.rotz_lacunarity = kwargs.get('rotz_lacunarity', 2.0)
        self.rotz_repeat = kwargs.get('rotz_repeat', 1024)

    #def sample_perlin(self, base, scale, x, min_val, max_val, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024):
    #    noise_val = self.perlin_noise.sample(base + x * scale, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    #    return noise_val * (max_val - min_val) + min_val


    def sample_perlin(self, frame_index, range_min, range_max, tremor_scale, octaves, persistence, lacunarity, repeat):
        # Prepare noise correctly with normalization
        t = frame_index / (self.frame_count - 1 if self.frame_count > 1 else 1)
        linear_value = (range_max - range_min) * t + range_min
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        interpolated_value = linear_value * noise_adjustment
        return interpolated_value

    def execute(self, **kwargs):

        if kwargs.__contains__("opac_perlin_settings"):
            perlin_settings = kwargs.pop("opac_perlin_settings")
            kwargs.update(perlin_settings)
            print("\033[1m\033[94mOPAC Perlin Settings applied!:\033[0m")

        # Process the input values
        self.process_kwargs(**kwargs)

        if not self.use_wiggle:
            return ([0] * self.frame_count,) * 8

        # More dynamic implementation this time
        zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z = (
            [self.sample_perlin(i, *param) for i in range(self.frame_count)]
            for param in [
                (self.zoom_range[0], self.zoom_range[1], self.zoom_tremor_scale, self.zoom_octaves, self.zoom_persistence, self.zoom_lacunarity, self.zoom_repeat),
                (self.angle_range[0], self.angle_range[1], self.angle_tremor_scale, self.angle_octaves, self.angle_persistence, self.angle_lacunarity, self.angle_repeat),
                (self.trx_range[0], self.trx_range[1], self.trx_tremor_scale, self.trx_octaves, self.trx_persistence, self.trx_lacunarity, self.trx_repeat),
                (self.try_range[0], self.try_range[1], self.try_tremor_scale, self.try_octaves, self.try_persistence, self.try_lacunarity, self.try_repeat),
                (self.trz_range[0], self.trz_range[1], self.trz_tremor_scale, self.trz_octaves, self.trz_persistence, self.trz_lacunarity, self.trz_repeat),
                (self.rotx_range[0], self.rotx_range[1], self.rotx_tremor_scale, self.rotx_octaves, self.rotx_persistence, self.rotx_lacunarity, self.rotx_repeat),
                (self.roty_range[0], self.roty_range[1], self.roty_tremor_scale, self.roty_octaves, self.roty_persistence, self.roty_lacunarity, self.roty_repeat),
                (self.rotz_range[0], self.rotz_range[1], self.rotz_tremor_scale, self.rotz_octaves, self.rotz_persistence, self.rotz_lacunarity, self.rotz_repeat)
            ]
        )
            
        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")

        log_curve("zoom", zoom)
        log_curve("angle", angle)
        log_curve("translation_x", translation_x)
        log_curve("translation_y", translation_y)
        log_curve("translation_z", translation_z)
        log_curve("rotation_3d_x", rotation_3d_x)
        log_curve("rotation_3d_y", rotation_3d_y)
        log_curve("rotation_3d_z", rotation_3d_z)

        print("")

        return zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z

class OPACPerlinSettings:
    """
        Configuration node for Perlin noise sampling in OPAC node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zoom_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "zoom_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "zoom_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "zoom_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "angle_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "angle_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "angle_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "angle_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "try_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "try_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "try_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "try_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "roty_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "roty_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "roty_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "roty_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("opac_perlin_settings",)
    FUNCTION = "process"
    CATEGORY = "SALT/Animation/Keyframing"

    def process(self, **kwargs):
        return (kwargs, )


class OPAC2Floats:
    """
        Converts a LIST input to FLOATS type
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "zoom": ("LIST", {}),
                "angle": ("LIST", {}),
                "translation_x": ("LIST", {}),
                "translation_y": ("LIST", {}),
                "translation_z": ("LIST", {}),
                "rotation_x": ("LIST", {}),
                "rotation_y": ("LIST", {}),
                "rotation_z": ("LIST", {})
            }
        }
    
    RETURN_TYPES = ("FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS")
    RETURN_NAMES = ("zoom", "angle", "translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z")
    FUNCTION = "convert"
    CATEGORY = "SALT/Animation/Keyframing"

    def convert(self, **kwargs):
        return (
                kwargs.get("zoom", [0]),
                kwargs.get("angle", [0]),
                kwargs.get("translation_x", [0]),
                kwargs.get("translation_y", [0]),
                kwargs.get("translation_z", [0]),
                kwargs.get("rotation_x", [0]),
                kwargs.get("rotation_y", [0]),
                kwargs.get("rotation_z", [0])
            )
    
class OPACListVariance:
    """
        Applies Perlin noise to each value in a list to create a OPAC Schedule out of it
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
            "optional": {
                "tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}), 
                "octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "opac_variance"
    CATEGORY = "SALT/Animation/Keyframing"

    #def sample_perlin(self, base, scale, x, min_val, max_val, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024):
    #    noise_val = self.perlin_noise.sample(base + x * scale, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    #    return noise_val * (max_val - min_val) + min_val
    
    def sample_perlin(self, frame_index, range_min, range_max, tremor_scale, octaves, persistence, lacunarity, repeat):
        # Prepare noise correctly with normalization
        t = frame_index / (self.frame_count - 1 if self.frame_count > 1 else 1)
        linear_value = (range_max - range_min) * t + range_min
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        interpolated_value = linear_value * noise_adjustment
        return interpolated_value

    def opac_variance(self, list_input, tremor_scale, octaves, persistence, lacunarity, repeat):
        self.frame_count = len(list_input) 
        varied_list = [
            self.sample_perlin(i, min(list_input), max(list_input), tremor_scale, octaves, persistence, lacunarity, self.frame_count)
            for i, _ in enumerate(self.frame_count)
        ]

        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("List Curve", varied_list)

        return (varied_list,)

    
class OPACList2ExecList:
    """
        Converts a list to a list output (iterative execution list)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "SALT/Animation/Keyframing"

    def convert(self, list_input):
        return (list_input, )
    
class OPACTransformImages:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "depth_maps": ("IMAGE", ),
                                
                "displacement_amplitude": ("FLOAT", {"min": 0, "max": 1024, "default": 60, "step": 0.1}),
                "displacement_angle_increment": ("INT", {"min": 1, "max": 24, "default": 3, "step": 1}),
                "displacement_start_frame": ("INT", {"min": 0, "max": 4096, "default": 0, "step": 1}),
                "displacement_end_frame": ("INT", {"min": 2, "max": 4096, "default": 60, "step": 1}),
                "displacement_preset": (list(movement_modes),),
                "displacement_easing": (list(easing_functions.keys()),),
                "displacement_tremor_scale": ("FLOAT", {"min": 0, "max": 100.0, "default": 0.02, "step": 0.01}),

                "edge_mode": (list(edge_modes.keys()),),

                "zoom_factor": ("FLOAT", {"min": 1, "max": 16, "default": 1, "step": 1}),
                "zoom_increment": ("FLOAT", {"min": 0, "max": 16, "default": 0, "step": 0.01}),
                "zoom_easing": (["ease-in", "ease-out", "ease-in-out", "bounce-in", "bounce-out", "bounce-in-out"],),
                "zoom_coordinate_x": ("INT", {"min": -1, "max": 8196, "default": -1, "step": 1}),
                "zoom_coordinate_y": ("INT", {"min": -1, "max": 8196, "default": -1, "step": 1}),
                "zoom_start_frame": ("INT", {"min": 0, "max": 4096, "default": 0, "step": 1}),
                "zoom_end_frame": ("INT", {"min": 2, "max": 4096, "default": 60, "step": 1}),
                "zoom_tremor_scale": ("FLOAT", {"min": 0, "max": 100.0, "default": 0.02, "step": 0.01}),

                "tremor_octaves": ("INT", {"min": 1, "max": 6, "default": 1}),
                "tremor_persistence": ("FLOAT", {"min": 0.01, "max": 1.0, "default": 0.5}),
                "tremor_lacunarity": ("FLOAT", {"min": 0.1, "max": 4.0, "default": 2, "step": 0.01}),

                "create_masks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    CATEGORY = "SALT/Animation/Transform"
    FUNCTION = "transform"

    def transform(
            self,
            images,
            depth_maps,
            displacement_amplitude,
            displacement_angle_increment,
            displacement_start_frame,
            displacement_end_frame,
            displacement_preset,
            displacement_easing,
            displacement_tremor_scale,
            edge_mode,
            zoom_factor,
            zoom_increment,
            zoom_easing,
            zoom_coordinate_x,
            zoom_coordinate_y,
            zoom_start_frame,
            zoom_end_frame,
            zoom_tremor_scale,
            tremor_octaves,
            tremor_persistence,
            tremor_lacunarity,
            create_masks,
    ):
        
        frames = []
        masks = []

        num_frames = images.shape[0]

        start_time = time.time()
        comfy_pbar = ProgressBar(num_frames)

        for frame_number in tqdm(range(num_frames), desc='Generating frames'):
            
            image = images[frame_number]
            depth_map = depth_maps[frame_number] if frame_number <= depth_maps.shape[0] else depth_maps[-1]
            frame, mask = generate_frame(
                num_frames=num_frames,
                frame_number=frame_number,
                amplitude=displacement_amplitude,
                angle_increment=displacement_angle_increment,
                movement_mode=displacement_preset,
                movement_tremor_scale=displacement_tremor_scale,
                easing_function=displacement_easing,
                texture_image=tensor2pil(image.unsqueeze(0)),
                displacement_image=tensor2pil(depth_map.unsqueeze(0)),
                edge_mode=edge_mode,
                zoom_factor=zoom_factor,
                zoom_increment=zoom_increment,
                zoom_easing=zoom_easing,
                zoom_coordinates=(zoom_coordinate_x, zoom_coordinate_y),
                create_mask=create_masks,
                repetition=False,
                zoom_start_frame=zoom_start_frame,
                zoom_end_frame=zoom_end_frame,
                zoom_tremor_scale=zoom_tremor_scale,
                displacement_start_frame=displacement_start_frame,
                displacement_end_frame=displacement_end_frame,
                tremor_octaves=tremor_octaves,
                tremor_lacunarity=tremor_lacunarity,
                tremor_persistence=tremor_persistence
            )

            if frame:
                frames.append(pil2tensor(frame.convert("RGB")))
                if create_masks and mask is not None:
                    masks.append(pil2mask(mask.convert("L")))

            comfy_pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        print("Frames generated.")
        print(f"Transform Animation Completed in {elapsed_time:.2f} seconds")

        return (torch.cat(frames, dim=0), torch.cat(masks, dim=0))
    

class OPCScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 60, "min": 1, "max": 4096}),
                "zoom_speed": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "pan_speed": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 5.0, "step": 0.001}),
                "pan_directions": ("STRING", {"default": "90,180,270"}),
                "direction_change_frames": ("STRING", {"default": "10,20,40"}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "direction_curve": (list(easing_functions.keys()), ),
                "start_x": ("FLOAT", {"default": 0, "min": -nodes.MAX_RESOLUTION, "max": nodes.MAX_RESOLUTION}),
                "start_y": ("FLOAT", {"default": 0}),
                "zoom_mode": (["zoom-in", "zoom-out", "zoom-in-out"], ),
                "layer_offsets": ("STRING", {"default": "1,0.8,0.6"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("float_layers",)
    FUNCTION = "execute"
    CATEGORY = "SALT/Animation/Keyframing/Parallax Motion"

    def __init__(self):
        self.scheduler = None

    def execute(self, **kwargs):

        self.process_kwargs(**kwargs)

        if not self.scheduler:
            raise Exception("Camera Scheduler was not initialized. Perhaps your settings are bugged?")
        
        mode = kwargs.get("zoom_mode", "zoom-in")
        layer_offsets = kwargs.get("layer_offsets", "1")

        animation_data = self.scheduler.animate(mode, layer_offsets)

        return (animation_data, )

    def process_kwargs(self, **kwargs):
        frame_count = kwargs.get("frame_count", 60)
        zoom_speed = kwargs.get("zoom_speed", 0.01)
        pan_speed = kwargs.get("pan_speed", 0.5)
        pan_directions = list(map(float, kwargs.get("pan_directions", "90,180,270").split(",")))
        direction_change_frames = list(map(int, kwargs.get("direction_change_frames", "10,20,40").split(",")))
        tremor_params = {
            "scale": kwargs.get("tremor_scale", 0.1),
            "octaves": kwargs.get("tremor_octaves", 1),
            "persistence": kwargs.get("tremor_persistence", 0.5),
            "lacunarity": kwargs.get("tremor_lacunarity", 2.0),
        }
        direction_curve = kwargs.get("direction_curve", "linear")
        start_x = kwargs.get("start_x", 0)
        start_y = kwargs.get("start_y", 0)

        self.scheduler = OrganicPerlinCameraScheduler(
            frame_count=frame_count,
            zoom_speed=zoom_speed,
            pan_speed=pan_speed,
            pan_directions=pan_directions,
            direction_change_frames=direction_change_frames,
            tremor_params=tremor_params,
            direction_curve=direction_curve,
            start_x=start_x,
            start_y=start_y,
        )

class OPCSLayerExtractor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_layers": ("LIST", ),
                "layer_index": ("INT", {"default": 0, "min": 0})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom_values", "x_values", "y_values")
    FUNCTION = "extract"
    CATEGORY = "SALT/Animation/Keyframing/Parallax Motion"

    def extract(self, **kwargs):
        animation_data = kwargs.get("float_layers", [])
        layer_index = kwargs.get("layer_index", 0)

        if layer_index >= len(animation_data):
            raise ValueError("Layer index out of range.")

        selected_layer_data = animation_data[layer_index]
        zoom_values = [frame[0] for frame in selected_layer_data]
        x_values = [frame[1] for frame in selected_layer_data]
        y_values = [frame[2] for frame in selected_layer_data]

        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("Zoom Values", zoom_values)
        log_curve("X Values", x_values)
        log_curve("Y Values", y_values)

        return zoom_values, x_values, y_values


class ParallaxMotion:
    """
    A node for generating min/max FLOAT values for Front and Back layers across X, Y, and Z axes
    in 2D plane translations to create a parallax effect, based on selected presets or custom input values.
    Adjusts these values based on the parallax intensity.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "zoom_preset": (["None", "Zoom In", "Zoom Out", "Zoom In/Out", "Zoom Out/In", "Custom"], {
                    "default": "None"
                }),
                "horizontal_pan_preset": (["None", "Pan Left → Right", "Pan Right → Left", "Pan Left → Center", 
                                           "Pan Right → Center", "Pan Center → Right", "Pan Center → Left", "Custom"], {
                    "default": "None"
                }),
                "vertical_pan_preset": (["None", "Pan Up → Down", "Pan Down → Up", "Pan Up → Center", 
                                         "Pan Down → Center", "Pan Center → Up", "Pan Center → Down", "Custom"], {
                    "default": "None"
                }),
                "custom_x_min": ("FLOAT", {"default": 0.0}),
                "custom_x_max": ("FLOAT", {"default": 0.0}),
                "custom_y_min": ("FLOAT", {"default": 0.0}),
                "custom_y_max": ("FLOAT", {"default": 0.0}),
                "custom_z_min": ("FLOAT", {"default": 1.0}),
                "custom_z_max": ("FLOAT", {"default": 1.0}),
                "parallax_intensity": ("FLOAT", {"default": 1.0}),
                "zoom_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("front_x_min", "front_x_max", "front_y_min", "front_y_max", "front_z_min", "front_z_max", "back_x_min", "back_x_max", "back_y_min", "back_y_max", "back_z_min", "back_z_max")
    FUNCTION = "generate_parameters"
    CATEGORY = "SALT/Animation/Keyframing/Parallax Motion"

    def generate_parameters(self, zoom_preset, horizontal_pan_preset, vertical_pan_preset, 
                        custom_x_min, custom_x_max, custom_y_min, custom_y_max, 
                        custom_z_min, custom_z_max, parallax_intensity, zoom_intensity):
        # Initialize default axis values
        x_min, x_max = 0, 0
        y_min, y_max = 0, 0
        z_min, z_max = 1, 1  # Default Z values assume no zoom (scale of 1)
        
        # Apply the selected zoom preset
        if zoom_preset in zoom_presets:
            z_min, z_max = zoom_presets[zoom_preset]
        
        # Apply the selected horizontal pan preset
        if horizontal_pan_preset in horizontal_pan_presets:
            x_min, x_max = horizontal_pan_presets[horizontal_pan_preset]
        
        # Apply the selected vertical pan preset
        if vertical_pan_preset in vertical_pan_presets:
            y_min, y_max = vertical_pan_presets[vertical_pan_preset]
        
        # For 'Custom' selections, override with custom values
        if zoom_preset == "Custom":
            z_min, z_max = custom_z_min, custom_z_max
        if horizontal_pan_preset == "Custom":
            x_min, x_max = custom_x_min, custom_x_max
        if vertical_pan_preset == "Custom":
            y_min, y_max = custom_y_min, custom_y_max

        # Calculate the back layer's values based on the parallax intensity
        back_x_min = x_min / parallax_intensity if x_min != 0 else 0
        back_x_max = x_max / parallax_intensity if x_max != 0 else 0
        back_y_min = y_min / parallax_intensity if y_min != 0 else 0
        back_y_max = y_max / parallax_intensity if y_max != 0 else 0
        # Z values are not adjusted by parallax intensity
        back_z_min, back_z_max = z_min, z_max

        # Adjust for zoom intensity for Z axis to increase front zoom relative to back with higher intensity values
        # This approach assumes the front layer's zoom can increase more than the back's as zoom_intensity increases.
        # Adjust the formula as needed to align with your specific vision for the effect.
        adjusted_front_z_min = z_min * (1 + zoom_intensity - 1)  # Example adjustment, enhances the front zoom based on intensity
        adjusted_front_z_max = z_max * (1 + zoom_intensity - 1)  # Similarly for max zoom

        # The back layer's zoom could remain unchanged, or you could apply a different formula if the back layer also needs adjustment.
        back_z_min, back_z_max = z_min, z_max  # Keeps back layer's zoom unchanged; modify as needed.


        return (x_min, x_max, y_min, y_max, adjusted_front_z_min, adjusted_front_z_max, back_x_min, back_x_max, back_y_min, back_y_max, back_z_min, back_z_max)


NODE_CLASS_MAPPINGS = {
    "OPAC": OPAC,
    "OPACPerlinSettings": OPACPerlinSettings,
    "OPAC2Floats": OPAC2Floats,
    "OPACListVariance": OPACListVariance,
    "OPACList2ExecList": OPACList2ExecList,
    #"OPACTransformImages": OPACTransformImages,
    "OPCScheduler": OPCScheduler,
    "OPCSLayerExtractor": OPCSLayerExtractor,
    "ParallaxMotion": ParallaxMotion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPAC": "OPAC Scheduler",
    "OPACPerlinSettings": "Keyframing Perlin Settings",
    "OPAC2Floats": "Keyframing Convert to FLOATS (MTB)",
    "OPACListVariance": "Apply Keyframing to List",
    "OPACList2ExecList": "Keyframing Convert Iterative Execution List",
    #"OPACTransformImages": "OPAC Transform Image Batch",
    "OPCScheduler": "Parallax Motion Camera Scheduler",
    "OPCSLayerExtractor": "Parallax Motion Camera Scheduler Extractor",
    "ParallaxMotion": "Parallax Motion Parameters Generator"
}