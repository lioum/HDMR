import sys
import cv2 as cv
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from hdmr_viewer.harris import compute_harris_response

def run_binary(bin_path: str, img_path: str) -> tuple[np.ndarray, np.ndarray]:
    result = subprocess.run([bin_path, img_path], capture_output=True, text=True)
    indices = np.array([tuple(int(i) for i in line.split(",")) for line in result.stdout.splitlines()], dtype=int)

    return indices[:, 0], indices[:, 1]

def ref_harris_points(img_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harris_response = compute_harris_response(img_gray)
    coord_flatten = np.argsort(-harris_response, axis=None)[:2000]

    return coord_flatten // img_gray.shape[1], coord_flatten % img_gray.shape[1]

def main():
    if len(sys.argv) < 2:
        exit(1)

    img_path = sys.argv[1]

    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    py_y, py_x = ref_harris_points(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    cpu_y, cpu_x = run_binary('build/hdmr', img_path)
    
    img[py_y, py_x] = [0, 255, 0]
    img[cpu_y, cpu_x] = [0, 255, 255]

    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
