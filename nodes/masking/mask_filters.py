from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import torch

from ... import logger
from SaltAI.modules.masking import MaskFilters
from SaltAI.modules.convert import pil2mask, mask2pil, pil2tensor
from SaltAI import NAME

def closest_odd(n):
    if n % 2 == 1:
        return n
    else:
        return n - 1

# MASK DOMINANT REGION

class SaltMaskDominantRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "threshold": ("INT", {"default":128, "min":0, "max":255, "step":1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "dominant_region"

    def dominant_region(self, masks, threshold=128):
        if not isinstance(threshold, list):
            threshold = [threshold]
        regions = []
        for i, mask in enumerate(masks):
            mask_pil = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.dominant_region(mask_pil, int(threshold[i if i < len(threshold) else -1]))
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

            
# MASK MINORITY REGION

class SaltMaskMinorityRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "threshold": ("INT", {"default":128, "min":0, "max":255, "step":1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "minority_region"

    def minority_region(self, masks, threshold=128):
        if not isinstance(threshold, list):
            threshold = [threshold]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.minority_region(pil_image, int(threshold[i if i < len(threshold) else -1]))
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

        
# MASK ARBITRARY REGION

class SaltMaskArbitaryRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "size": ("INT", {"default":256, "min":1, "max":4096, "step":1}),
                        "threshold": ("INT", {"default":128, "min":0, "max":255, "step":1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "arbitrary_region"

    def arbitrary_region(self, masks, size=256, threshold=128):
        if not isinstance(threshold, list):
            threshold = [threshold]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.arbitrary_region(pil_image, size, int(threshold[i if i < len(threshold) else -1]))
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
        

# MASK SMOOTH REGION

class SaltMaskSmoothRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "sigma": ("FLOAT", {"default":5.0, "min":0.0, "max":128.0, "step":0.1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "smooth_region"

    def smooth_region(self, masks, sigma=128):
        if not isinstance(sigma, list):
            sigma = [sigma]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.smooth_region(pil_image, sigma[i if i < len(sigma) else -1])
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

        
# MASK ERODE REGION

class SaltMaskErodeRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "iterations": ("INT", {"default":5, "min":1, "max":64, "step":1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "erode_region"

    def erode_region(self, masks, iterations=5):
        if not isinstance(iterations, list):
            iterations = [iterations]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.erode_region(pil_image, iterations[i if i < len(iterations) else -1])
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

          
# MASKS SUBTRACT
          
class SaltMaskSubtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
        return (subtracted_masks,)
        
# MASKS ADD
          
class SaltMaskAdd:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "add_masks"

    def add_masks(self, masks_a, masks_b):
        if masks_a.shape != masks_b.shape or len(masks_a.shape) != 3 or len(masks_b.shape) != 3:
            raise ValueError("Both input tensors must be of shape [N, H, W].")
        added_masks = torch.clamp(masks_a + masks_b, 0, 255)
        return (added_masks,)  
        
# MASKS Invert
          
class SaltMaskInvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "add_masks"

    def add_masks(self, masks):
        return (1. - masks,)
        
# MASK DILATE REGION

class SaltMaskDilateRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "iterations": ("INT", {"default":5, "min":1, "max":64, "step":1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "dilate_region"

    def dilate_region(self, masks, iterations=5):
        if not isinstance(iterations, list):
            iterations = [iterations]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.dilate_region(pil_image, iterations[i if i < len(iterations) else -1])
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    
        
# MASK FILL REGION

class SaltMaskFillRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "fill_region"

    def fill_region(self, masks):
        regions = []
        for mask in masks:
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.fill_region(pil_image)
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

        
# MASK THRESHOLD

class SaltMaskThresholdRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                        "black_threshold": ("INT",{"default":75, "min":0, "max": 255, "step": 1}),
                        "white_threshold": ("INT",{"default":175, "min":0, "max": 255, "step": 1}),
                    }
                }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "threshold_region"

    def threshold_region(self, masks, black_threshold=75, white_threshold=255):
        if not isinstance(black_threshold, list):
            black_threshold = [black_threshold]
        if not isinstance(white_threshold, list):
            white_threshold = [white_threshold]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.threshold_region(
                pil_image, 
                int(black_threshold[i if i < len(black_threshold) else -1]), 
                int(white_threshold[i if i < len(white_threshold) else -1])
            )
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
        
# MASK FLOOR REGION

class SaltMaskFloorRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "floor_region"

    def floor_region(self, masks):
        regions = []
        for mask in masks:
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.floor_region(pil_image)
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)


# MASK CEILING REGION

class SaltMaskCeilingRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "ceiling_region"
    
    def ceiling_region(self, masks):
        regions = []
        for mask in masks:
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.ceiling_region(pil_image)
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

        
# MASK GAUSSIAN REGION

class SaltMaskGaussianRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1024, "step": 0.1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "gaussian_region"

    def gaussian_region(self, masks, radius=5.0):
        if not isinstance(radius, list):
            radius = [radius]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            region_mask = MaskFilters.gaussian_region(pil_image, radius[i if i < len(radius) else -1])
            region_tensor = pil2mask(region_mask)
            regions.append(region_tensor)
        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskEdgeDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "method": (["sobel", "canny"], ),
                "low_threshold": ("INT", {"default": 50, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 150, "min": 0, "max": 255, "step": 1}),
                "sobel_ksize": ("INT", {"default": 5, "min": 1, "max": 7, "step": 1})
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "edge_detection"

    def edge_detection(self, masks, method='canny', low_threshold=50, high_threshold=150, sobel_ksize=5):
        regions = []

        if not isinstance(low_threshold, list):
            low_threshold = [low_threshold]
        if not isinstance(high_threshold, list):
            high_threshold = [high_threshold]
        if not isinstance(sobel_ksize, list):
            sobel_ksize = [sobel_ksize]

        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            image_array = np.array(pil_image.convert('L'))

            if method == 'canny':
                edges = cv2.Canny(
                    image_array, 
                    low_threshold[i if i < len(low_threshold) else -1], 
                    high_threshold[i if i < len(high_threshold) else -1]
                )
            elif method == 'sobel':
                sobelx = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=sobel_ksize[i if i < len(sobel_ksize) else -1])
                sobely = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=sobel_ksize[i if i < len(sobel_ksize) else -1])
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(255 * edges / np.max(edges))
            else:
                raise ValueError(f"Invalid edge detection mode '{method}', please use sobel, or canny.")
            
            edge_pil = Image.fromarray(edges)

            region_tensor = pil2mask(edge_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskGradientRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "gradient_region"

    def gradient_region(self, masks, kernel_size=3):
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            image_array = np.array(pil_image.convert('L'))

            current_kernel_size = kernel_size[i if i < len(kernel_size) else -1]
            kernel = np.ones((current_kernel_size, current_kernel_size), np.uint8)

            gradient = cv2.morphologyEx(image_array, cv2.MORPH_GRADIENT, kernel)
            gradient_pil = Image.fromarray(gradient)

            region_tensor = pil2mask(gradient_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMorphoOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "operation": (["opening", "closing"], ),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "morphological_oper"

    def morphological_oper(self, masks, operation='opening', kernel_size=3, iterations=1):
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]
        if not isinstance(iterations, list):
            iterations = [iterations]
        
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            image_array = np.array(pil_image.convert('L'))

            current_kernel_size = kernel_size[i if i < len(kernel_size) else -1]
            current_iterations = iterations[i if i < len(iterations) else -1]

            kernel = np.ones((current_kernel_size, current_kernel_size), np.uint8)
            
            if operation == 'opening':
                region = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel, iterations=current_iterations)
            elif operation == 'closing':
                region = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel, iterations=current_iterations)
            else:
                errmsg = f"Invalid operation '{operation}'. Use 'opening' or 'closing'."
                logger.error(errmsg)
                raise ValueError(errmsg)

            region_pil = Image.fromarray(region)

            region_tensor = pil2mask(region_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskAdaptiveThresholdingRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "block_size": ("INT", {"default": 11, "min": 3, "max": 255, "step": 2}),
                "constant": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "adaptive_thresholding"

    def adaptive_thresholding(self, masks, block_size=11, constant=2):
        if not isinstance(block_size, list):
            block_size = [block_size]
        if not isinstance(constant, list):
            constant = [constant]

        block_size = [closest_odd(val) for val in block_size]
        
        regions = []
        for i, mask in enumerate(masks):
            pil_image = mask2pil(mask.unsqueeze(0))
            image_array = np.array(pil_image.convert('L'))
            
            current_block_size = block_size[i if i < len(block_size) else -1]
            current_C = constant[i if i < len(constant) else -1]
            
            adaptive_thresh = cv2.adaptiveThreshold(image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, current_block_size, current_C)

            adaptive_pil = ImageOps.invert(Image.fromarray(adaptive_thresh))
            region_tensor = pil2mask(adaptive_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)


class SaltMaskHistogramEqualizationRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "histogram_equalization"

    def histogram_equalization(self, masks):
        regions = []
        for mask in masks:
            pil_image = mask2pil(mask.unsqueeze(0))
            image_array = np.array(pil_image.convert('L'))            
            equalized = cv2.equalizeHist(image_array)
            equalized_pil = ImageOps.invert(Image.fromarray(equalized))
            region_tensor = pil2mask(equalized_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskRegionLabeling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "region_labeling"

    def region_labeling(self, masks, threshold=[128]):

        if not isinstance(threshold, list):
            threshold = [threshold]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image.convert('L'))

            _, thresh_image = cv2.threshold(image_array, threshold[i if i < len(threshold) else -1], 255, cv2.THRESH_BINARY)

            num_labels, labels_im = cv2.connectedComponents(thresh_image)
            max_label = max(num_labels - 1, 1)

            hues = np.linspace(0, 179, num=max_label + 1, endpoint=False, dtype=np.uint8)

            label_hue = np.zeros_like(labels_im, dtype=np.uint8)
            for i in range(1, num_labels):
                label_hue[labels_im == i] = hues[i]

            saturation = np.uint8(np.where(labels_im == 0, 0, 255))
            value = np.uint8(np.where(labels_im == 0, 0, 255))

            labeled_img = cv2.merge([label_hue, saturation, value])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            labeled_pil = Image.fromarray(labeled_img)
            region_tensor = pil2tensor(labeled_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)



class SaltMaskContourExtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "thresholds": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "contour_extraction"

    def contour_extraction(self, masks, thresholds=[50, 100, 150, 200]):
        regions = []

        if not isinstance(thresholds, list):
            thresholds = [thresholds]

        for mask in masks:
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image.convert('L'))

            combined_contours = np.zeros_like(image_array)

            for threshold in thresholds:
                _, thresh_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined_contours, contours, -1, (255, 255, 255), 1)

            contour_pil = Image.fromarray(combined_contours)
            region_tensor = pil2mask(contour_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)

    

class SaltMaskBilateralFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "diameter": ("INT", {"default": 9, "min": 1, "max": 31, "step": 1}),
                "sigmaColor": ("FLOAT", {"default": 75.0, "min": 0.0, "max": 200.0, "step": 0.1}),
                "sigmaSpace": ("FLOAT", {"default": 75.0, "min": 0.0, "max": 200.0, "step": 0.1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "bilateral_filter"

    def bilateral_filter(self, masks, diameter=9, sigmaColor=75.0, sigmaSpace=75.0):
        if not isinstance(diameter, list):
            diameter = [diameter]
        if not isinstance(sigmaColor, list):
            sigmaColor = [sigmaColor]
        if not isinstance(sigmaSpace, list):
            sigmaSpace = [sigmaSpace]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image.convert('RGB'))

            current_diameter = diameter[i if i < len(diameter) else -1]
            current_sigmaColor = sigmaColor[i if i < len(sigmaColor) else -1]
            current_sigmaSpace = sigmaSpace[i if i < len(sigmaSpace) else -1]

            filtered = cv2.bilateralFilter(image_array, current_diameter, current_sigmaColor, current_sigmaSpace)

            filtered_pil = Image.fromarray(filtered)
            region_tensor = pil2mask(filtered_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskClipHardeningFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 6.0, "step": 0.01}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "sharpening_filter"

    def sharpening_filter(self, masks, strength=1.5):
        if not isinstance(strength, list):
            strength = [strength]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image.convert('RGB'))

            current_strength = strength[i if i < len(strength) else -1]

            kernel = np.array([[-1, -1, -1],
                               [-1, 8 * current_strength, -1],
                               [-1, -1, -1]])

            sharpened = cv2.filter2D(image_array, -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            sharpened_pil = Image.fromarray(sharpened)
            region_tensor = pil2mask(sharpened_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)


class SaltMaskSharpeningFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "strength": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "sharpening_filter"

    def sharpening_filter(self, masks, strength=1.5):
        if not isinstance(strength, list):
            strength = [strength]

        strength = [int(val) for val in strength]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))

            for _ in range(strength[i if i < len(strength) else -1]):
                pil_image = pil_image.filter(ImageFilter.SHARPEN)

            region_tensor = pil2mask(pil_image)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    
    

class SaltMaskSkeletonization:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "strength": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1})
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "skeletonization"

    def skeletonization(self, masks, iterations=[1], strength=[1]):
        if not isinstance(iterations, list):
            iterations = [iterations]
        if not isinstance(strength, list):
            strength = [strength]

        iterations = [int(val) for val in iterations]
        strength = [int(val) for val in strength]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image.convert('L'))

            skeleton = np.zeros_like(image_array)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = image_array
                for _ in range(iterations[i if i < len(iterations) else -1]):
                    eroded = cv2.erode(eroded, element)

                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(image_array, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                image_array = eroded.copy()

                if cv2.countNonZero(image_array) == 0:
                    break

                for _ in range(strength[i if i < len(strength) else -1]):
                    image_array = image_array + image_array

            skeleton_pil = Image.fromarray(skeleton)
            region_tensor = pil2mask(skeleton_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskNoiseAddition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "mean": ("FLOAT", {"default": 0.0, "min": -255.0, "max": 255.0, "step": 0.1}),
                "stddev": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "noise_addition"

    def noise_addition(self, masks, mean=0.0, stddev=25.0):
        if not isinstance(mean, list):
            mean = [mean]
        if not isinstance(stddev, list):
            stddev = [stddev]

        regions = []
        for i, mask in enumerate(masks):
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            image_array = np.array(pil_image)

            current_mean = mean[i if i < len(mean) else -1]
            current_stddev = stddev[i if i < len(stddev) else -1]

            noise = np.random.normal(current_mean, current_stddev, image_array.shape)
            noisy_image = image_array + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

            noisy_pil = Image.fromarray(noisy_image)
            region_tensor = pil2mask(noisy_pil)
            regions.append(region_tensor)

        regions_tensor = torch.cat(regions, dim=0)
        return (regions_tensor,)
    

class SaltMaskRegionSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }

    CATEGORY = f"{NAME}/Masking/Filter"

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("region1", "region2", "region3", "region4", "region5", "region6")

    FUNCTION = "isolate_regions"

    def isolate_regions(self, masks):
        region_outputs = []

        for mask in masks:
            pil_image = ImageOps.invert(mask2pil(mask.unsqueeze(0)))
            mask_array = np.array(pil_image.convert('L'))

            num_labels, labels_im = cv2.connectedComponents(mask_array)

            outputs = [np.zeros_like(mask_array) for _ in range(6)]

            for i in range(1, min(num_labels, 7)):
                outputs[i-1][labels_im == i] = 255

            for output in outputs:
                output_pil = Image.fromarray(output)
                region_tensor = pil2mask(output_pil)
                region_outputs.append(region_tensor)

        regions_tensor = torch.stack(region_outputs, dim=0).view(len(masks), 6, *mask.size())
        return tuple(regions_tensor.unbind(dim=1))
    

NODE_DISPLAY_NAME_MAPPINGS = {
    'SaltMaskDominantRegion': 'Dominant Mask Regions',
    'SaltMaskMinorityRegion': 'Minority Mask Regions',
    'SaltMaskArbitaryRegion': 'Arbitrary Mask Regions',
    'SaltMaskSmoothRegion': 'Smooth Mask Regions',
    'SaltMaskErodeRegion': 'Erode Mask Regions',
    'SaltMaskSubtract': 'Subtract Mask Regions',
    'SaltMaskAdd': 'Add Mask Regions',
    'SaltMaskInvert': 'Invert Mask Regions',
    'SaltMaskDilateRegion': 'Dilate Mask Regions',
    'SaltMaskFillRegion': 'Fill Mask Regions',
    'SaltMaskThresholdRegion': 'Threshold Mask Regions',
    'SaltMaskFloorRegion': 'Floor Mask Regions',
    'SaltMaskCeilingRegion': 'Ceiling Mask Regions',
    'SaltMaskGaussianRegion': 'Gaussian Mask Regions',
    'SaltMaskEdgeDetection': 'Edge Detect Mask Regions',
    'SaltMaskGradientRegion': 'Gradient Filter Mask Regions',
    'SaltMaskAdaptiveThresholdingRegion': 'Adaptive Threshold Mask Regions',
    'SaltMaskHistogramEqualizationRegion': 'Hisogram Equalize Mask Regions',
    'SaltMaskRegionLabeling': 'Label Mask Regions (RGB)',
    'SaltMaskContourExtraction': 'Countour Mask Regions Extraction',
    'SaltMaskBilateralFilter': 'Bilateral Filter Mask Regions',
    'SaltMaskClipHardeningFilter': 'Clip Harden Region',
    'SaltMaskSharpeningFilter': 'Sharpen Mask Regions',
    'SaltMaskSkeletonization': 'Skeletonize Mask Regions',
    'SaltMaskNoiseAddition': 'Add Noise to Mask Regions',
    'SaltMaskRegionSplit': 'Split Regions',
}

NODE_CLASS_MAPPINGS = {
    key: globals()[key] for key in NODE_DISPLAY_NAME_MAPPINGS.keys()
}