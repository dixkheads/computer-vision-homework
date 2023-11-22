import numpy as np
import cv2
from proj2_code.student_harris import get_gradients


def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.

    """
    magnitudes = np.sqrt(dx ** 2 + dy ** 2)
    orientations = np.arctan2(dy, dx)
    return magnitudes, orientations


def get_feat_vec(x, y, magnitudes, orientations, feature_width):
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.


    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    A useful function to look at would be np.histogram.
    """

    # Initialize the feature vector
    fv = np.zeros(128)

    # Calculate the 4x4 grid size
    cell_size = feature_width // 4

    # Define bin centers for the histogram
    bin_centers = np.arange(-7 * np.pi / 8, 9 * np.pi / 8, np.pi / 4)

    # Loop through the grid cells
    for i in range(4):
        for j in range(4):
            # Calculate the cell boundaries
            cell_x_start = int(x - feature_width // 2 + i * cell_size)
            cell_x_end = int(x - feature_width // 2 + (i + 1) * cell_size)
            cell_y_start = int(y - feature_width // 2 + j * cell_size)
            cell_y_end = int(y - feature_width // 2 + (j + 1) * cell_size)

            # Initialize an 8-bin histogram for this cell
            cell_histogram = np.zeros(8)

            # Loop through the pixels in the cell
            for pixel_x in range(cell_x_start, cell_x_end):
                for pixel_y in range(cell_y_start, cell_y_end):
                    # Calculate the orientation bin for the pixel
                    pixel_orientation = orientations[pixel_y, pixel_x]
                    bin_index = int((pixel_orientation - (-7 * np.pi / 8)) / (np.pi / 4))

                    # Update the cell histogram
                    cell_histogram[bin_index] += magnitudes[pixel_y, pixel_x]

            # Append the cell histogram to the feature vector
            fv[i * 32 + j * 8:i * 32 + (j + 1) * 8] = cell_histogram

    # Normalize the feature vector
    fv /= np.linalg.norm(fv)

    # Apply the power correction (raise to the power of 0.9)
    fv = fv ** 0.9

    return fv


def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before.

    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'

    # Get image gradients
    dx, dy = get_gradients(image)

    # Initialize the array to store feature vectors
    num_points = len(x)
    feat_dim = 128  # SIFT-like feature dimensionality
    fvs = np.zeros((num_points, feat_dim))

    # Loop through the interest points and compute features
    for i in range(num_points):
        # Calculate feature vector at the current interest point
        fv = get_feat_vec(x[i], y[i], dx, dy, feature_width)

        # Store the feature vector in the result array
        fvs[i] = fv

    return fvs