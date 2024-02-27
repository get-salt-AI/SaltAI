from PIL import Image
import numpy as np
import torch
import re

try:

    def sanitize_filename(filename):
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        filename = re.sub(r'[\n\r\t]', '', filename)
        filename = filename.strip()
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
        if filename.upper() in reserved_names:
            filename += '_'
        max_length = 255
        if len(filename) > max_length:
            root, ext = re.match(r'^(.*?)(\.[^.]+)?$', filename).groups()
            filename = root[:max_length-len(ext)-1] + '_' + ext if ext else root[:max_length]
        return filename

except Exception as e:
    print("There is an error in this file!")
    raise e