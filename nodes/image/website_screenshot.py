from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time
import os

from nodes import MAX_RESOLUTION

from ...modules.convert import pil2tensor
from ... import NAME, logger

class SaltWebsiteScreenshot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "full_page_screenshot": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"min": 64, "max": MAX_RESOLUTION, "default": 1920,}),
                "height": ("INT", {"min": 64, "max": MAX_RESOLUTION, "default": 1080}),
                "scroll_to": ("INT", {"min": 0, "max": 99999}),
                "retries": ("INT", {"min": 1, "max": 6, "default": 3}),
                "wait_time": ("INT", {"min": 1, "max": 60, "default": 1})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "url")

    FUNCTION = "screenshot"
    CATEGORY = f"{NAME}/Image/Misc"

    def screenshot(self, url, width=1920, height=1080, full_page_screenshot=False, scroll_to=0, retries=3, wait_time=2):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--start-maximized")

        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        driver.set_window_size(width, height)

        pil_image = None

        try:
            for attempt in range(retries):
                try:
                    logger.info(f"Attempting to open webpage: {url}")
                    driver.get(url)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == retries - 1:
                        raise
            
            if full_page_screenshot:
                logger.info("Taking full page screenshot...")
                total_width = driver.execute_script("return document.body.scrollWidth")
                total_height = driver.execute_script("return document.body.scrollHeight")
                driver.set_window_size(total_width, total_height)
            else:
                logger.info(f"Taking {width}x{height} screenshot...")
                driver.set_window_size(width, height)
                if scroll_to > 0:
                    driver.execute_script(f"window.scrollTo(0, {scroll_to})")

            time.sleep(wait_time)

            temp_path = "temp_screenshot.png"
            driver.save_screenshot(temp_path)
            logger.info(f"Temporary screenshot saved to {temp_path}")

            with Image.open(temp_path) as img:
                pil_image = img.convert("RGB")
            
            os.remove(temp_path)
            logger.info(f"Temporary file {temp_path} deleted")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        finally:
            driver.quit()
        
        return (pil2tensor(pil_image), url)
    

NODE_CLASS_MAPPINGS = {
    "SaltWebsiteScreenshot": SaltWebsiteScreenshot
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltWebsiteScreenshot": "Simple Website Screenshot"
}