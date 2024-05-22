import cv2
from PIL import Image
import numpy as np
import torch

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

def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil2cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def image2mask(x, release=False):
    z = x.clone()
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 1, 1, 3)
    w = w.to(z.device)
    g = torch.sum(z * w, dim=-1)
    if release:
        del x
    return g

def masks2pils(x, dtype=np.uint8, release=False):
    if len(x.shape) != 3:
        raise ValueError("Input tensor must be of shape [N, H, W].")
    N, H, W = x.shape
    masks = []
    for n in range(N):
        # Check if the tensor values are in the range [0, 1] and convert to [0, 255] if necessary
        if x.max() <= 1.0:
            mask_numpy = (x[n].cpu().detach().numpy() * 255).astype(dtype)
        else:
            mask_numpy = x[n].cpu().detach().numpy().astype(dtype)
        
        mask = Image.fromarray(mask_numpy, 'L')
        masks.append(mask)
    if release:
        del x  # Release the tensor from memory if specified
    return masks