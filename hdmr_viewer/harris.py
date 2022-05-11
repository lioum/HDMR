import numpy as np
from typing import Tuple
from scipy import signal

def gauss_kernel(size: int, sizey: int=None) -> np.ndarray:
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    g = np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    return g

def gauss_derivative_kernels(size: int, sizey: int=None) -> Tuple[np.ndarray, np.ndarray]:
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    gx = - x * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    gy = - y * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))

    return gx,gy

def gauss_derivatives(im: np.ndarray, size: int, sizey: int=None) -> Tuple[np.ndarray, np.ndarray]:
    gx,gy = gauss_derivative_kernels(size, sizey=sizey)

    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')

    return imx,imy

def compute_harris_response(image, derivative_ker_size: int = 3, opening_size: int = 3):
    
    imx,imy = gauss_derivatives(image, derivative_ker_size)

    gauss = gauss_kernel(opening_size) # opening param

    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / (Wtr + 1)  # 1 seems to be a reasonable value for epsilon
