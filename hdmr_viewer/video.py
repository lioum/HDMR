import sys
import os
from typing import Iterator
import cv2 as cv
import shutil
from pathlib import Path
import numpy as np
import subprocess

def main():
    video_path = sys.argv[1]

    out_dir_str = "output_images"
    in_dir_str = "video_images"
    in_dir = Path(in_dir_str)
    out_dir = Path(out_dir_str)

    shutil.rmtree(in_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)

    in_dir.mkdir()
    out_dir.mkdir()

    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", "fps=30", in_dir_str + "/out%04d.jpg"]
        , text=True)

    result.check_returncode()

    images_path = list(Path("video_images").iterdir())
    images_path.sort(key=Path.as_posix)

    print(len(images_path))

    for img_path in images_path:
        out_path = out_dir / img_path.name
        result = subprocess.run(["release/hdmr_gpu", img_path.as_posix(), out_path.as_posix()])
        result.check_returncode()

    print("Finished processing images")

    subprocess.run(['ffmpeg', '-r', '30', '-i', out_dir_str + '/out%04d.jpg'
                    , '-c:v', 'libx264', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', 'out.mp4'])
    
if __name__ == "__main__":
    main()
