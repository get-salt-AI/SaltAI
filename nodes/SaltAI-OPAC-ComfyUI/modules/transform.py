import math
import random
import traceback
import cv2
from PIL import Image
import numpy as np
import torch

# Simple perlin noise based on Power Noise Suite
class PerlinNoise:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 10000)
        torch.manual_seed(self.seed)
        self.p = torch.randperm(256, dtype=torch.int32)
        self.p = torch.cat((self.p, self.p))

    def fade(self, t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x):
        h = hash & 15
        grad = 1 + (h & 7)
        if h & 8:
            grad = -grad
        return grad * x

    def noise(self, x):
        x = x % 255
        X = torch.tensor(x, dtype=torch.float32).floor().long()
        x -= X.float()
        u = self.fade(x)

        A = self.p[X % 256]
        B = self.p[(X + 1) % 256]

        return self.lerp(u, self.grad(self.p[A], x), self.grad(self.p[B], x - 1))

    def sample(self, x, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += self.noise(x * scale * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return (total / max_value).item()

# Helper function
def apply_noise(perlin_noise, frame_number, scale, octaves, persistence, lacunarity):
    return perlin_noise.sample(frame_number, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

# Movement Presets
def movement_mode_curves(perlin_noise, movement_tremor_scale, frame_number, tremor_octaves, tremor_persistence, tremor_lacunarity):
    return {
        'orbit': lambda angle_increment, amplitude: (
            amplitude * math.cos(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'side-to-side': lambda angle_increment, amplitude: (
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            0
        ),
        'up-and-down': lambda angle_increment, amplitude: (
            0,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'diagonal_bottom_left': lambda angle_increment, amplitude: (
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'diagonal_top_right': lambda angle_increment, amplitude: (
            -amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            -amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        )
    }

movement_modes = ["orbit", "side-to-side", "up-and-down", "diagonal_bottom_left", "diagonal_top_right"]

# Easing Functions
def bounce_in(t):
    return 1 - bounce_out(1 - t)

def bounce_out(t):
    if t < 4/11.0:
        return (121 * t * t) / 16.0
    elif t < 8/11.0:
        return (363/40.0 * t * t) - (99/10.0 * t) + 17/5.0
    elif t < 9/10.0:
        return (4356/361.0 * t * t) - (35442/1805.0 * t) + 16061/1805.0
    else:
        return (54/5.0 * t * t) - (513/25.0 * t) + 268/25.0

def bounce_in_out(t):
    if t < 0.5:
        return 0.5 * bounce_in(t*2)
    else:
        return 0.5 * bounce_out((t - 0.5)*2) + 0.5

easing_functions = {
    'ease-in': lambda t: t ** 3,
    'ease-out': lambda t: 1 - (1 - t) ** 3,
    'ease-in-out': lambda t: 4 * t * (1 - t),
    'bounce-in': bounce_in,
    'bounce-out': bounce_out,
    'bounce-in-out': bounce_in_out
}

# Edge Handling Modes
edge_modes = {
    'clamp': lambda idx, max_val: np.clip(idx, 0, max_val - 1),
    'mirror': lambda idx, max_val: np.abs(2*(idx // max_val) * (idx % max_val) + idx % max_val),
    'wrap': lambda idx, max_val: np.mod(idx, max_val),
    'smear': lambda idx, max_val: np.where(idx < 0, 0, np.where(idx >= max_val, max_val - 1, idx))
}

# Edge Detection
def edge_fx(indices, max_value, mode, mask=None, repetition=False):
    original_indices = np.copy(indices)
    
    if mode in edge_modes:
        indices = edge_modes[mode](indices, max_value)

    if repetition:
        indices = np.mod(indices, max_value)
        
    if mask is not None:
        mask[np.where(indices != original_indices)] = 255

    return indices

def seam_fx(texture, mask, edge_mode):
    rows, cols, _ = texture.shape
    new_texture = np.copy(texture)
    
    for row in range(rows):
        for col in range(cols):
            if mask[row, col] == 128:
                neighbor_indices = [(i, j) for i in range(row-1, row+2) for j in range(col-1, col+2)
                                    if 0 <= i < rows and 0 <= j < cols]
                new_texture[row, col] = np.mean([texture[i, j] for i, j in neighbor_indices], axis=0).astype(np.uint8)
                
    return edge_fx(new_texture, 255, edge_mode)

def generate_frame(
        num_frames, 
        frame_number,
        texture_image, 
        displacement_image, 
        amplitude, 
        angle_increment=3, 
        displacement_start_frame=None,
        displacement_end_frame=None,
        movement_mode=None, 
        movement_tremor_scale=0.01,
        easing_function=None,
        edge_mode="clamp", 
        repetition=False,
        create_mask=False,
        zoom_factor=1,
        zoom_increment=0.0,
        zoom_easing="ease-in-out",
        zoom_coordinates=(-1, -1),
        zoom_start_frame=None,
        zoom_end_frame=None,
        zoom_tremor_scale=0.01,
        tremor_octaves=1,
        tremor_persistence=0.5,
        tremor_lacunarity=2,
        seed=1492
    ):
    perlin_noise = PerlinNoise(seed=seed)

    modes = movement_mode_curves(perlin_noise, movement_tremor_scale, frame_number, tremor_octaves, tremor_persistence, tremor_lacunarity)
    current_mode = modes[movement_mode]

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    if zoom_start_frame is None:
        zoom_start_frame = 0
    elif zoom_start_frame >= zoom_end_frame:
        zoom_start_frame = 0

    if zoom_end_frame is None:
        zoom_end_frame = num_frames
    elif zoom_end_frame <= zoom_start_frame:
        zoom_end_frame = num_frames

    if displacement_start_frame is None:
        displacement_start_frame = 0
    elif displacement_start_frame >= displacement_end_frame:
        displacement_start_frame = 0

    if displacement_end_frame is None:
        displacement_end_frame = num_frames
    elif displacement_end_frame <= displacement_start_frame:
        displacement_end_frame = num_frames
    
    if zoom_increment != 0:
        if zoom_start_frame <= frame_number <= zoom_end_frame:
            relative_frame = frame_number - zoom_start_frame
            total_zoom_frames = zoom_end_frame - zoom_start_frame
            zoom_progress = relative_frame / total_zoom_frames

            if zoom_easing != 'None':
                dynamic_zoom_factor = 1 + easing_functions[zoom_easing](zoom_progress) * zoom_increment
            else:
                dynamic_zoom_factor = 1 + zoom_progress * zoom_increment

            dynamic_zoom_factor *= 1 + perlin_noise.sample(frame_number, scale=zoom_tremor_scale, octaves=tremor_octaves, persistence=tremor_persistence, lacunarity=tremor_lacunarity) * 0.01
        else:
            dynamic_zoom_factor = 1
    else:
        dynamic_zoom_factor = zoom_factor

    if dynamic_zoom_factor != 1:
        texture_image = zoom_and_crop(np.array(texture_image)[:, :, ::-1], dynamic_zoom_factor, zoom_coordinates)
        displacement_image = zoom_and_crop(np.array(displacement_image.convert("L")), dynamic_zoom_factor, zoom_coordinates)
    else:
        texture_image = np.array(texture_image)[:, :, ::-1]
        displacement_image = np.array(displacement_image.convert("L"))

    try:
        texture_rows, texture_cols, _ = texture_image.shape
        if displacement_image.shape[:2] != (texture_rows, texture_cols):
            displacement_image = cv2.resize(displacement_image, (texture_cols, texture_rows))

        if displacement_start_frame <= frame_number <= displacement_end_frame:
            if movement_mode:
                x_displacement, y_displacement = current_mode(angle_increment, amplitude)
                if easing_function:
                    relative_frame = frame_number - displacement_start_frame
                    total_displacement_frames = displacement_end_frame - displacement_start_frame
                    ease_value = easing_functions[easing_function](relative_frame / total_displacement_frames)
                    x_displacement *= ease_value
                    y_displacement *= ease_value
            else:
                angle = frame_number * angle_increment
                x_displacement = amplitude * math.cos(math.radians(angle))
                y_displacement = amplitude * math.sin(math.radians(angle))
        else:
            x_displacement, y_displacement = 0, 0

        texture_rows, texture_cols, _ = texture_image.shape
        y_idx, x_idx = np.indices((texture_rows, texture_cols))

        occupancy = np.zeros((texture_rows, texture_cols), dtype=np.uint8)
        edge_mask = np.zeros((texture_rows, texture_cols), dtype=np.uint8)

        x_idx = x_idx.astype(np.float32) + (x_displacement * displacement_image / 255.0).astype(np.float32)
        y_idx = y_idx.astype(np.float32) + (y_displacement * displacement_image / 255.0).astype(np.float32)

        x_idx = edge_fx(x_idx, texture_cols, edge_mode, mask=edge_mask, repetition=repetition)
        y_idx = edge_fx(y_idx, texture_rows, edge_mode, mask=edge_mask, repetition=repetition)

        new_y_idx = np.clip(y_idx.astype(int), 0, texture_rows - 1)
        new_x_idx = np.clip(x_idx.astype(int), 0, texture_cols - 1)

        occupancy[new_y_idx, new_x_idx] = 1

        #left_behind_mask = np.where(occupancy == 0, 128, 0).astype(np.uint8)
        #combined_mask = cv2.bitwise_or(edge_mask, left_behind_mask)

        transformed_texture = texture_image[new_y_idx, new_x_idx]

        otexture_zoomed = texture_image
        if dynamic_zoom_factor != 1:
            transformed_texture = zoom_and_crop(transformed_texture, dynamic_zoom_factor, zoom_coordinates)
            otexture_zoomed = zoom_and_crop(texture_image, dynamic_zoom_factor, zoom_coordinates)

        inverse_displacement_mask = np.where(displacement_image == 0, 1, 0).astype(np.uint8)

        original_non_focal = cv2.bitwise_and(otexture_zoomed, otexture_zoomed, mask=inverse_displacement_mask)
        transformed_non_focal = cv2.bitwise_and(transformed_texture, transformed_texture, mask=inverse_displacement_mask)

        original_non_focal_gray = cv2.cvtColor(original_non_focal, cv2.COLOR_BGR2GRAY)
        transformed_non_focal_gray = cv2.cvtColor(transformed_non_focal, cv2.COLOR_BGR2GRAY)

        diff_mask = cv2.absdiff(original_non_focal_gray, transformed_non_focal_gray)

        diff_mask_blurred = cv2.GaussianBlur(diff_mask, (0, 0), 3.0)
        diff_mask_clamped = np.interp(diff_mask_blurred, [30, 35], [0, 255]).astype(np.uint8)

        transformed_texture_pil = Image.fromarray(cv2.cvtColor(transformed_texture, cv2.COLOR_BGR2RGB))
        mask_image_pil = None
        if create_mask:
            mask_image_pil = Image.fromarray(diff_mask_clamped)

        return transformed_texture_pil, mask_image_pil
            
    except Exception as e:
        traceback.print_exc()
        return None, None

def zoom_and_crop(image, zoom_factor, coordinates=(-1, -1)):
    if zoom_factor == 1:
        return image

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    x, y = coordinates

    if x == -1:
        x = width // 2
    if y == -1:
        y = height // 2

    x = int(x * zoom_factor)
    y = int(y * zoom_factor)

    y1 = max(y - height // 2, 0)
    y2 = min(y1 + height, new_height)
    x1 = max(x - width // 2, 0)
    x2 = min(x1 + width, new_width)

    if y2 == new_height:
        y1 = new_height - height
    if x2 == new_width:
        x1 = new_width - width

    cropped_image = resized_image[y1:y2, x1:x2]
    
    return cropped_image
