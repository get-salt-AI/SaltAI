from PIL import Image, ImageOps, ImageFilter
from skimage.measure import label, regionprops
import numpy as np

from SaltAI import NAME

class MaskFilters():

    @staticmethod
    def crop_dominant_region(image, padding=0):
        from scipy.ndimage import label
        grayscale_image = image.convert("L")
        binary_image = grayscale_image.point(lambda x: 255 if x > 128 else 0, mode="1")
        labeled_image, num_labels = label(np.array(binary_image))
        largest_label = max(range(1, num_labels + 1), key=lambda i: np.sum(labeled_image == i))
        largest_region_mask = (labeled_image == largest_label).astype(np.uint8) * 255
        bbox = Image.fromarray(largest_region_mask, mode="L").getbbox()
        cropped_image = image.crop(bbox)
        size = max(cropped_image.size)
        padded_size = size + 2 * padding
        centered_crop = Image.new("L", (padded_size, padded_size), color="black")
        left = (padded_size - cropped_image.width) // 2
        top = (padded_size - cropped_image.height) // 2
        centered_crop.paste(cropped_image, (left, top), mask=cropped_image)

        return ImageOps.invert(centered_crop)

    @staticmethod
    def crop_minority_region(image, padding=0):
        from scipy.ndimage import label
        grayscale_image = image.convert("L")
        binary_image = grayscale_image.point(lambda x: 255 if x > 128 else 0, mode="1")
        labeled_image, num_labels = label(np.array(binary_image))
        smallest_label = min(range(1, num_labels + 1), key=lambda i: np.sum(labeled_image == i))
        smallest_region_mask = (labeled_image == smallest_label).astype(np.uint8) * 255
        bbox = Image.fromarray(smallest_region_mask, mode="L").getbbox()
        cropped_image = image.crop(bbox)
        size = max(cropped_image.size)
        padded_size = size + 2 * padding
        centered_crop = Image.new("L", (padded_size, padded_size), color="black")
        left = (padded_size - cropped_image.width) // 2
        top = (padded_size - cropped_image.height) // 2
        centered_crop.paste(cropped_image, (left, top), mask=cropped_image)

        return ImageOps.invert(centered_crop)
                
    @staticmethod
    def crop_region(image, region_type="dominant", padding=0):
        grayscale_image = image.convert("L")
        binary_image = grayscale_image.point(lambda x: 255 if x > 128 else 0, mode="1")
        labeled_image = label(np.array(binary_image))
        regions = regionprops(labeled_image)

        if not regions:
            return image, (image.size, (0, 0, 0, 0))

        if region_type == "minority":
            target_region = min(regions, key=lambda r: r.area)
        else:  # "dominant"
            target_region = max(regions, key=lambda r: r.area)

        minr, minc, maxr, maxc = target_region.bbox
        width = maxc - minc
        height = maxr - minr
        side_length = max(width, height) + 2 * padding
        
        x_center = (minc + maxc) // 2
        y_center = (minr + maxr) // 2
        new_minr = y_center - side_length // 2
        new_maxr = y_center + side_length // 2
        new_minc = x_center - side_length // 2
        new_maxc = x_center + side_length // 2

        new_minc = max(new_minc, 0)
        new_minr = max(new_minr, 0)
        new_maxc = min(new_maxc, image.width)
        new_maxr = min(new_maxr, image.height)

        cropped_image = image.crop((new_minc, new_minr, new_maxc, new_maxr))

        crop_data = (cropped_image.size, (new_minc, new_minr, new_maxc, new_maxr))
        return cropped_image, crop_data
        
    @staticmethod
    def dominant_region(image, threshold=128):
        from scipy.ndimage import label
        image = ImageOps.invert(image.convert("L"))
        binary_image = image.point(lambda x: 255 if x > threshold else 0, mode="1")
        region, n = label(np.array(binary_image))
        sizes = np.bincount(region.flatten())
        dominant = 0
        try:
            dominant = np.argmax(sizes[1:]) + 1
        except ValueError:
            pass
        dominant_region_mask = (region == dominant).astype(np.uint8) * 255
        result = Image.fromarray(dominant_region_mask, mode="L")
        return result.convert("RGB")

    @staticmethod
    def minority_region(image, threshold=128):
        from scipy.ndimage import label
        grayscale_image = image.convert("L")
        binary_image = grayscale_image.point(lambda x: 255 if x > threshold else 0, mode="1")
        binary_array = np.array(binary_image)
        labeled_array, num_features = label(binary_array)
        sizes = np.bincount(labeled_array.flatten())
        sizes[0] = 0
        smallest_region = np.argmin(sizes) if np.any(sizes) else 0
        smallest_region_mask = (labeled_array == smallest_region).astype(np.uint8)
        inverted_mask = Image.fromarray(smallest_region_mask * 255, mode="L")
        rgb_image = Image.merge("RGB", [inverted_mask, inverted_mask, inverted_mask])

        return rgb_image

    @staticmethod
    def arbitrary_region(image, size, threshold=128):
        from skimage.measure import label, regionprops
        image = image.convert("L")
        binary_image = image.point(lambda x: 255 if x > threshold else 0, mode="1")
        labeled_image = label(np.array(binary_image))
        regions = regionprops(labeled_image)

        image_area = binary_image.size[0] * binary_image.size[1]
        scaled_size = size * image_area / 10000

        filtered_regions = [region for region in regions if region.area >= scaled_size]
        if len(filtered_regions) > 0:
            filtered_regions.sort(key=lambda region: region.area)
            smallest_region = filtered_regions[0]
            region_mask = (labeled_image == smallest_region.label).astype(np.uint8) * 255
            result = Image.fromarray(region_mask, mode="L")
            return ImageOps.invert(result)

        return ImageOps.invert(image)
        
    @staticmethod
    def smooth_region(image, tolerance):
        from scipy.ndimage import gaussian_filter
        image = image.convert("L")
        mask_array = np.array(image)
        smoothed_array = gaussian_filter(mask_array, sigma=tolerance)
        threshold = np.max(smoothed_array) / 2
        smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
        smoothed_image = Image.fromarray(smoothed_mask, mode="L")
        return ImageOps.invert(smoothed_image.convert("RGB"))

    @staticmethod
    def erode_region(image, iterations=1):
        from scipy.ndimage import binary_erosion
        image = ImageOps.invert(image.convert("L"))
        binary_mask = np.array(image) > 0
        eroded_mask = binary_erosion(binary_mask, iterations=iterations)
        eroded_image = Image.fromarray(eroded_mask.astype(np.uint8) * 255, mode="L")
        return eroded_image.convert("RGB")

    @staticmethod
    def dilate_region(image, iterations=1):
        from scipy.ndimage import binary_dilation
        image = ImageOps.invert(image.convert("L"))
        binary_mask = np.array(image) > 0
        dilated_mask = binary_dilation(binary_mask, iterations=iterations)
        dilated_image = Image.fromarray(dilated_mask.astype(np.uint8) * 255, mode="L")
        return dilated_image.convert("RGB")
    
    @staticmethod
    def fill_region(image):
        from scipy.ndimage import binary_fill_holes
        image = ImageOps.invert(image.convert("L"))
        binary_mask = np.array(image) > 0
        filled_mask = binary_fill_holes(binary_mask)
        filled_image = Image.fromarray((filled_mask * 255).astype(np.uint8), mode="L")
        return filled_image

    @staticmethod
    def combine_masks(*masks):
        if len(masks) < 1:
            raise ValueError(f"\033[34m{NAME}\033[0m Error: At least one mask must be provided.")
        dimensions = masks[0].size
        for mask in masks:
            if mask.size != dimensions:
                raise ValueError(f"\033[34m{NAME}\033[0m Error: All masks must have the same dimensions.")

        inverted_masks = [mask.convert("L") for mask in masks]
        combined_mask = Image.new("L", dimensions, 255)
        for mask in inverted_masks:
            combined_mask = Image.fromarray(np.minimum(np.array(combined_mask), np.array(mask)), mode="L")

        return combined_mask

    @staticmethod
    def threshold_region(image, black_threshold=0, white_threshold=255):
        gray_image = image.convert("L")
        mask_array = np.array(gray_image)
        mask_array[mask_array < black_threshold] = 0
        mask_array[mask_array > white_threshold] = 255
        thresholded_image = Image.fromarray(mask_array, mode="L")
        return ImageOps.invert(thresholded_image)
        
    @staticmethod
    def floor_region(image):
        gray_image = image.convert("L")
        mask_array = np.array(gray_image)
        non_black_pixels = mask_array[mask_array > 0]
        
        if non_black_pixels.size > 0:
            threshold_value = non_black_pixels.min()
            mask_array[mask_array > threshold_value] = 255  # Set whites to 255
            mask_array[mask_array <= threshold_value] = 0  # Set blacks to 0
        
        thresholded_image = Image.fromarray(mask_array, mode="L")
        return ImageOps.invert(thresholded_image)    
        
    @staticmethod
    def ceiling_region(image, offset=30):
        if offset < 0:
            offset = 0
        elif offset > 255:
            offset = 255
        grayscale_image = image.convert("L")
        mask_array = np.array(grayscale_image)
        mask_array[mask_array < 255 - offset] = 0
        mask_array[mask_array >= 250] = 255
        filtered_image = Image.fromarray(mask_array, mode="L")
        return ImageOps.invert(filtered_image)
        
    @staticmethod
    def gaussian_region(image, radius=5.0):
        image = ImageOps.invert(image.convert("L"))
        image = image.filter(ImageFilter.GaussianBlur(radius=int(radius)))
        return image.convert("RGB")