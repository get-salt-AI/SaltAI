import torch
import numpy as np
from PIL import Image

def tensor2pil(x):
    return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def pil2tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)

def mask2pil(x):
    x = 1. - x
    if x.ndim != 3:
        print(f"Expected a 3D tensor ([N, H, W]). Got {x.ndim} dimensions.")
        x = x.unsqueeze(0) 
    x_np = x.cpu().numpy()
    if x_np.ndim != 3:
        x_np = np.expand_dims(x_np, axis=0) 
    return Image.fromarray(np.clip(255. * x_np[0, :, :], 0, 255).astype(np.uint8), 'L')

def pil2mask(x):
    if x.mode == 'RGB':
        r, g, b = x.split()
        x = Image.fromarray(np.uint8(0.2989 * np.array(r) + 0.5870 * np.array(g) + 0.1140 * np.array(b)), 'L')
    elif x.mode != 'L':
        raise ValueError("Unsupported image mode, expected 'RGB' or 'L', got {}".format(x.mode))
    mask = torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)
    return mask