from PIL import Image
import numpy as np
import torch
import re

try:

    def tensor2pil(x):
        return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
    def pil2tensor(x):
        return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)

    def pil2mask(x):
        if x.mode == 'RGB':
            r, g, b = x.split()
            x = Image.fromarray(np.uint8(0.2989 * np.array(r) + 0.5870 * np.array(g) + 0.1140 * np.array(b)), 'L')
        elif x.mode != 'L':
            raise ValueError("Unsupported image mode, expected 'RGB' or 'L', got {}".format(x.mode))
        mask = torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)
        return mask
    

except Exception as e:
    print("There is an error in this file!")
    raise e