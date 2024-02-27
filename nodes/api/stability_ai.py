import torch

from ...modules.convert import pil2tensor, tensor2pil

import base64
import http.client
import mimetypes
import os
import json
import time
import threading
import sys
import itertools

import cv2
import torch
import io
from PIL import Image
import uuid

import folder_paths

MAIN_CACHE = folder_paths.get_temp_directory()

API = "api.stability.ai"
MODELS = [
    'SVD',
]
ENDPOINTS = {
    # Generation
    'svd': '/v2alpha/generation/image-to-video',
    # Results
    'svd_result': '/v2alpha/generation/image-to-video/result/'
}

class SAIAPI:
    def __init__(self, api_key):
        self.conn = http.client.HTTPSConnection(API)
        self.api_key = api_key
        self.boundary = "---011000010111000001101001"
        self.headers = {
            'authorization': f"Bearer {self.api_key}",
            'content-type': f"multipart/form-data; boundary={self.boundary}"
        }

    def create_payload(self, **kwargs):
        print("Creating payload...")

        payload_parts = []
        for key, value in kwargs.items():
            if key == 'image':
                if isinstance(value, str):
                    with open(value, 'rb') as file:
                        image_data = file.read()
                    content_type = mimetypes.guess_type(value)[0] or 'image/png'
                    filename = os.path.basename(value)
                elif isinstance(value, Image.Image):
                    img_byte_arr = io.BytesIO()
                    image_format = value.format or 'PNG'
                    value.save(img_byte_arr, format=image_format)
                    image_data = img_byte_arr.getvalue()
                    content_type = 'image/png' if image_format == 'PNG' else 'image/jpeg'
                    filename = f'image.{image_format.lower()}'
                else:
                    raise ValueError("image must be a string (file path) or a PIL Image")

                part = (
                    f"--{self.boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode('utf-8') + image_data
            elif key == 'seed':
                seed = SAIAPI.trim_seed(value)
                part = (
                    f"--{self.boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"seed\"\r\n\r\n"
                    f"{seed}\r\n"
                ).encode('utf-8')
            else:
                part = (
                    f"--{self.boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                    f"{value}\r\n"
                ).encode('utf-8')

            payload_parts.append(part)

        payload = b"\r\n".join(payload_parts) + f"\r\n--{self.boundary}--\r\n".encode('utf-8')
        print("Payload created.")
        return payload

    def svd_img2vid(self, image, seed=0, cfg=2.5, mbi=42):

        animation_started = False

        payload = self.create_payload(image=image, seed=seed, cfg_scale=cfg, motion_bucket_id=mbi)
        print("Contacting Stability AI API...")
        self.conn.request("POST", ENDPOINTS['svd'], payload, self.headers)
        res = self.conn.getresponse()
        if res.status != 200:
            print(f"API request failed with status code {res.status}: {res.reason}")
        data = res.read()
        response = json.loads(data.decode("utf-8"))

        id = response.get('id')
        if not id:
            # Poll until ID becomes available or a timeout/error occurs
            attempts = 0
            max_attempts = 10
            while not id and attempts < max_attempts:
                time.sleep(5)
                attempts += 1
                self.conn.request("POST", ENDPOINTS['svd'], payload, self.headers)
                res = self.conn.getresponse()
                data = res.read()
                response = json.loads(data.decode("utf-8"))
                id = response.get('id')

            if not id:
                print("Failed to obtain result ID from Stability AI API.")
                return None

        print(f"Inference started, ID: {id}")

        ret_header = self.headers
        ret_header['content-type'] = f'application/json; type=video/mp4; boundary={self.boundary}'
        while True:
            self.conn.request("GET", ENDPOINTS['svd_result'] + id, headers=ret_header)
            res = self.conn.getresponse()
            result_data = res.read()
            status_code = res.status

            try:
                result = json.loads(result_data.decode("utf-8"))

                if 'finishReason' in result:

                    if result['finishReason'] == 'SUCCESS':
                        from pprint import pprint
                        pprint(result, indent=4)
                        video_data = base64.b64decode(result['video'])
                        frames = self.mp4_to_pil_frames(video_data)
                        print("Video generated and received.")
                        print(f"Seed used for the video: {result.get('seed', 'Not available')}")
                        return frames
                    elif status_code in [400, 404, 500]:
                        result = json.loads(result_data.decode("utf-8"))
                        print(f"Error: {result.get('name')}")
                        for error in result.get('errors', []):
                            print(error)
                        break
                    else:
                        print(f"Unexpected status code received: {status_code}")
                        break

                else:
                    if not animation_started:
                        animation_started = True
                    if animation_started:
                        print("Still waiting...")
                    time.sleep(10)

            except UnicodeDecodeError: # We're always getting back binary despite trying to get back JSON, why?
                frames = self.get_frames(result_data)
                print("Video generated and received.")
                return frames

    def get_frames(self, video_data):
        temp_cache = os.path.join(MAIN_CACHE, "downloads")
        os.makedirs(temp_cache, exist_ok=True)
        temp_file = os.path.join(temp_cache, f"{uuid.uuid4()}.mp4")
        with open(temp_file, 'wb') as file:
            file.write(video_data)

        vidcap = cv2.VideoCapture(temp_file)
        success, frame = vidcap.read()
        frames = []

        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            success, frame = vidcap.read()

        vidcap.release()
        os.remove(temp_file)
        return frames
    
    @staticmethod
    def trim_seed(value, max_value=2147483648):
        if value > max_value:
            return max_value
        return value

class SaltAIStableVideoDiffusion:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "mbi": ("INT", {"min": 1, "max": 255, "default": 40, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)

    FUNCTION = "generate_video"
    CATEGORY = f"SALT/API/Stability API"

    def generate_video(self, image, api_key, seed=0, cfg=2.5, mbi=40):

        if api_key.strip() == "":
            raise Exception("A Stability AI API Key is required for video generaiton.")

        try:
            api = SAIAPI(api_key)

            if image.dim() != 4 and image.size(0) != 1:
                raise Exception("Only one image is allowed for Stability AI Stable Video Generation API.")
            
            image = tensor2pil(image)
            if image.size not in [(1024, 576), (576, 1024), (768, 768)]:
                raise Exception("Image resolution can only be within the following sizes: 1024x576, 576x1024, 768x768")
            
            frames = api.svd_img2vid(image=image, seed=seed, cfg=cfg, mbi=mbi)
            if frames and len(frames) > 0:
                frame_tensors = [pil2tensor(frame) for frame in frames]
                frame_batch = torch.cat(frame_tensors, dim=0)
            else:
                raise Exception("No frames found from SVD video temp file.")
        except Exception as e:
            raise e
        
        return (frame_batch, )
    
    
NODE_CLASS_MAPPINGS = {
    "SaltAIStableVideoDiffusion": SaltAIStableVideoDiffusion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltAIStableVideoDiffusion": "Stable Video Diffusion"
}