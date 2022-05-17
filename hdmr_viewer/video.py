import sys
import os
import shutil
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


def process_images(img_dir: Path, out_dir: Path):
    images_path = list(img_dir.iterdir())
    images_path.sort(key=Path.as_posix)

    def process_img(img_path: Path):
        out_path = out_dir / img_path.name
        result = subprocess.run(["release/hdmr_gpu", img_path.as_posix(), out_path.as_posix()])
        result.check_returncode()

    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {executor.submit(process_img, img): img for img in images_path}
        for future in concurrent.futures.as_completed(tasks):
            try:
                _ = future.result()
            except Exception as e:
                print(e)

    print("Finished processing images")

def main():
    video_path = sys.argv[1]

    out_dir_str = "output_images"
    in_dir_str = "video_images"
    in_dir = Path(in_dir_str)
    out_dir = Path(out_dir_str)

    shutil.rmtree(in_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)

    if os.path.exists("out.mp4"):
        os.remove("out.mp4")

    in_dir.mkdir()
    out_dir.mkdir()

    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", "fps=30", in_dir_str + "/out%04d.jpg"]
        , text=True)

    result.check_returncode()

    process_images(in_dir, out_dir)

    result = subprocess.run(['ffmpeg', '-r', '30', '-i', out_dir_str + '/out%04d.jpg'
                    , '-c:v', 'libx264', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', 'out.mp4'])
    
    result.check_returncode()

if __name__ == "__main__":
    main()
