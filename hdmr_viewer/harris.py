import cv2 as cv
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

def bubble2maskeroded(img_gray: np.ndarray, border: int=10) -> np.ndarray:
    """
    Returns the eroded mask of a given image, to remove pixels which are close to the border.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    
    Returns
    -------
    mask: np.array of shape (rows, cols) and dtype bool
        Image mask.
    """
    if img_gray.ndim > 2:
        raise ValueError(
            """bubble2maskeroded: img_gray must be a grayscale image.
            The image you passed has %d dimensions instead of 2.
            Try to convert it to grayscale before passing it to bubble2maskeroded.
            """ % (img_gray.ndim, ))
    mask = img_gray > 0
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (border*2,border*2))
    # new: added a little closing below because some bubbles have some black pixels inside
    mask_er = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3,3)))
    mask_er = cv.erode(mask.astype(np.uint8), 
                        kernel, 
                        borderType=cv.BORDER_CONSTANT, 
                        borderValue=0)
    return mask_er > 0

# TODO complete this function
def detect_harris_points(image_gray: np.ndarray, max_keypoints: int=30, 
                         min_distance: int=25, threshold: float=0.1) -> np.ndarray:
    """
    Detects and returns a sorted list of coordinates for each corner keypoint detected in an image.
    
    Parameters
    ----------
    image_gray: np.array
        Input image
    max_keypoints: int, default=30
        Number of keypoints to return, at most (we may have less keypoints)
    min_distance: int, default=25
        Minimum distance between two keypoints
    threshold: float, default=0.1
        For each keypoint k_i, we ensure that its response h_i will verify
        $h_i > min(response) + threshold * (max(reponse) - min(response))$
    
    Returns
    -------
    corner_coord: np.array of shape (N, 2) and dtype int
        Array of corner keypoint 2D coordinates, with N <= max_keypoints
    """
    # 1. Compute Harris corner response
    harris_resp = compute_harris_response(image_gray)
    
    # 2. Filtering
    # 2.0 Mask init: all our filtering is performed using a mask
    detect_mask = np.ones(harris_resp.shape, dtype=bool)

    # 2.2 Response threshold
    h_min = harris_resp.min()
    detect_mask &= harris_resp > h_min + threshold*(harris_resp.max() - h_min)
    # 2.3 Non-maximal suppression
    dil = cv.dilate(harris_resp, cv.getStructuringElement(cv.MORPH_ELLIPSE, (min_distance, min_distance)))
    detect_mask &= np.isclose(dil, harris_resp)  # keep only local maximas
               
    # 3. Select, sort and filter candidates
    # get coordinates of candidates
    candidates_coords = np.transpose(detect_mask.nonzero())
    # ...and their values
    candidate_values = harris_resp[detect_mask]
    #sort candidates
    sorted_indices = np.argsort(-candidate_values)
    # keep only the bests
    best_corners_coordinates = candidates_coords[sorted_indices][:max_keypoints]

    return best_corners_coordinates
