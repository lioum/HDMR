import sys
import cv2 as cv
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from hdmr_viewer.harris import detect_harris_points

def run_binary(bin_path: str, img_path: str) -> np.ndarray:
    result = subprocess.run([bin_path, img_path], capture_output=True, text=True)
    indices = np.array([tuple(int(i) for i in line.split(",")[:2]) for line in result.stdout.splitlines()], dtype=int)

    return indices

def main():
    if len(sys.argv) < 2:
        exit(1)

    img_path = sys.argv[1]

    nb_points = 25
    if len(sys.argv) > 2:
        nb_points = int(sys.argv[2]) or nb_points

    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    cpu = run_binary("build/hdmr", img_path)
    python = detect_harris_points(cv.cvtColor(img, cv.COLOR_RGB2GRAY), max_keypoints=nb_points)
    
    plt.imshow(img)
    plt.scatter(python[:nb_points,1], python[:nb_points, 0], s=6, c='red', alpha=0.5)
    plt.scatter(cpu[:nb_points, 1], cpu[:nb_points, 0], s=6, c='cyan', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
