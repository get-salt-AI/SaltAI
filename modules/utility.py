import subprocess
import shutil

# Resolve FFMPEG, idea borrowed from VideoHelperSuite
# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True, capture_output=True).stdout.decode("utf-8")
    except Exception as e:
        print(f"Error checking ffmpeg version at {path}: {e}")
        return 0
    score = 0
    # Rough layout of the importance of various features
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3), ("svtav1", 5), ("libopus", 1)]
    for criterion in simple_criterion:
        if criterion[0] in version:
            score += criterion[1]
    # Obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

def find_ffmpeg():
    ffmpeg_paths = []
    # Attempt to use imageio_ffmpeg if available
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_paths.append(get_ffmpeg_exe())
    except ImportError:
        print("imageio_ffmpeg is not available, trying system ffmpeg")

    # Check for system ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        ffmpeg_paths.append(system_ffmpeg)

    if not ffmpeg_paths:
        print("No valid ffmpeg found.")
        return None

    # Select the ffmpeg path with the highest suitability score
    ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)
    if ffmpeg_path:
        print(f"Using ffmpeg at {ffmpeg_path}")
    return ffmpeg_path


# Exports

ffmpeg_path = find_ffmpeg()