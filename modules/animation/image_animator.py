import os
import subprocess

import tempfile
import imageio
import numpy as np
from PIL import Image

from SaltAI.modules.convert import tensor2pil

#############################
# BSD 2-Clause License
#
# Copyright (c) 2019, imageio
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################

class ImageAnimator:
    def __init__(self, tensor, fps=10, quality="high"):
        self.tensor = tensor
        self.fps = int(fps)
        self.quality = (quality or "")
        self.frame_delay = int(1000 / fps)

    def _prepare_images(self):
        return [tensor2pil(img) for img in self.tensor]
    
    def _optimize_palette(self, images):
        if self.quality.lower() == "high":
            rgb_images = [img.convert("RGB") for img in images]
            quantized_images = [img.quantize(colors=256, method=Image.MEDIANCUT) for img in rgb_images]
            return quantized_images
        else:
            return images

    def _ffmpeg_metadata(self, path, metadata):
        temp_path = path + "_temp.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-i", path,
            "-metadata", f"author={metadata['author']}",
            "-metadata", f"comment={metadata['website']}",
            "-codec", "copy",
            temp_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(temp_path, path)

    def save_animation(self, filename, format="GIF", audio=None):
        images = self._prepare_images()

        if self.quality.lower() == "high" and format.upper() == "GIF":
            optimized_images = self._optimize_palette(images)
        else:
            optimized_images = images
        
        video_frames = [np.array(img.convert("RGB")) for img in optimized_images]
        
        codec_options = {
            "MP4": {"codec": "libx264"},
            "WEBM": {"codec": "libvpx", "bitrate": "1M"},
            "AVI": {"codec": "mpeg4"}
        }

        if format.upper() in codec_options:
            temp_video_path = filename + "_temp." + format.lower()
            imageio.mimsave(temp_video_path, video_frames, format=format.lower(), fps=self.fps, **codec_options[format.upper()])
            
            if audio:
                # Save the audio bytes buffer to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                    audio_file.write(audio)
                    audio_file_path = audio_file.name
                
                # Combine video and audio using FFmpeg
                command = [
                    "ffmpeg",
                    "-y",
                    "-i", temp_video_path,
                    "-i", audio_file_path,
                    "-c:v", codec_options[format.upper()]['codec'],
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-shortest",
                    filename
                ]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                os.remove(temp_video_path)  # Clean up temporary video file
                os.remove(audio_file_path)  # Clean up temporary audio file
            else:
                # Move temporary video to final path if no audio
                os.rename(temp_video_path, filename)

        elif format.upper() == "GIF":
            optimized_images[0].save(filename, save_all=True, append_images=optimized_images[1:], optimize=False, duration=self.frame_delay, loop=0)
        elif format.upper() == "WEBP":
            imageio.mimsave(filename, video_frames, format='webp', fps=self.fps)
        else:
            raise ValueError("Unsupported format. Please choose 'GIF', 'WEBP', 'MP4', 'WEBM', or 'AVI'.")

        # Add metadata for applicable formats
        if format.upper() not in ["WEBP", "GIF", "WEBM"]:
            self._ffmpeg_metadata(filename, {"author": "Salt.AI", "website": "http://getsalt.ai"})