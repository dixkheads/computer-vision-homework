import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.

    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """

    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(kernel, kernel)

    return kernel


def my_filter2D(image, filt, bias=0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
    -   filt: filter that will be used in the convolution
    -   bias: bias term (optional)

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = np.zeros_like(image)  # Initialize the output image with zeros

    # Get filter dimensions
    filt_height, filt_width = filt.shape
    image_height, image_width = image.shape[:2]

    # Calculate padding size
    pad_height = filt_height // 2
    pad_width = filt_width // 2

    # Pad the image with zeros
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)

    # Perform 2D convolution
    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            conv_image[i - pad_height, j - pad_width] = np.sum(region * filt) + bias

    return conv_image


def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy using a sobel filter.
    Sobel filters can be used to approximate the image gradient

    Helpful functions: my_filter2D from above

    Args:
    -   image: A numpy array of shape (m,n) containing the image


    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """

    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    # Define Sobel kernels for computing image gradients
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Convolve the image with Sobel kernels to compute gradients
    ix = my_filter2D(image, sobel_x)
    iy = my_filter2D(image, sobel_y)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return ix, iy


def remove_border_vals(image, x, y, c, window_size=16):
    """
    Remove interest points that are too close to a border to allow SIFT feature extraction.
    Make sure you remove all points where a window around that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale or color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove (e.g., 16).
        Treat the center point of this window as the bottom right of the center-most 4 pixels.

    Returns:
    -   x: A numpy array of shape (N-border_vals_removed,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-border_vals_removed,) containing y-coordinates of interest points
    -   c: numpy array of shape (N-border_vals_removed,) containing the confidences of each pixel
    """

    border_vals_removed = []  # List to store indices of points to keep

    # Image dimensions
    image_height, image_width = image.shape[:2]

    for i in range(min(len(x), len(y))):
        # Check if a window of size window_size can be formed around each point
        if (x[i] - window_size // 2 >= 0 and
                x[i] + window_size // 2 < image_width and
                y[i] - window_size // 2 >= 0 and
                y[i] + window_size // 2 < image_height):
            border_vals_removed.append(i)

    # Filter x, y, and c based on the points that passed the window size check
    x = x[border_vals_removed]
    y = y[border_vals_removed]
    c = c[border_vals_removed]

    return x, y, c


def second_moments(ix, iy, ksize=7, sigma=10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a Gaussian filter.

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of Gaussian filter (e.g., 7)
    -   sigma: standard deviation of Gaussian filter (e.g., 10)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    # Generate a 2D Gaussian kernel using the provided ksize and sigma
    gaussian_kernel = get_gaussian_kernel(ksize, sigma)

    # Compute the second moments using the Gaussian filter
    sx2 = my_filter2D(ix * ix, gaussian_kernel)
    sy2 = my_filter2D(iy * iy, gaussian_kernel)
    sxsy = my_filter2D(ix * iy, gaussian_kernel)

    return sx2, sy2, sxsy


def corner_response(sx2, sy2, sxsy, alpha):
    """
    Given second moments, calculate corner response.
    R = det(M) - alpha * (trace(M))^2
    where M = [[sx2, sxsy],
               [sxsy, sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in the Corner Response equation (e.g., 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    # Calculate the determinant and trace of the M matrix
    det_M = sx2 * sy2 - sxsy * sxsy
    trace_M = sx2 + sy2

    # Calculate the corner response R
    R = det_M - alpha * (trace_M ** 2)

    return R



    """
    Implement non-maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non-zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Args:
    -   R: numpy nd-array of shape (m, n)
    -   neighborhood_size: int that is the size of the neighborhood to find local maxima (e.g., 7)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero
    """

    # Use maximum_filter to find local maxima in a neighborhood
    local_maxima = maximum_filter(R, size=neighborhood_size)

    # Keep values in R that are equal to the local maxima and above the median of R
    R_local_pts = (R == local_maxima) & (R >= np.median(R))

    return R_local_pts


def get_interest_points(image, n_pts=1500):
    """
    Implement the Harris corner detector to start with.

    Args:
    -   image: A numpy array of shape (m, n, c),
                image may be grayscale or color (your choice)
    -   n_pts: integer of the number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m, n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """

    x, y, R_local_pts, confidences = None, None, None, None

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################

    # Compute image gradients
    ix, iy = get_gradients(image)

    # Compute second moments
    sx2, sy2, sxsy = second_moments(ix, iy)

    # Calculate corner response scores
    R = corner_response(sx2, sy2, sxsy, alpha=0.05)

    # Perform non-maximum suppression
    R_local_pts = non_max_suppression(R, neighborhood_size=7)

    # Remove interest points near the border
    x, y, confidences = remove_border_vals(image, np.arange(image.shape[1]), np.arange(image.shape[0]), R)

    # Sort and select the top n_pts interest points
    sorted_indices = np.argsort(R[y, x])[::-1]
    x = x[sorted_indices[:n_pts]]
    y = y[sorted_indices[:n_pts]]
    confidences = R[y, x]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return x, y, R_local_pts, confidences



