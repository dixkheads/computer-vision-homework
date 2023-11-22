import numpy as np
from sklearn.neighbors import KDTree


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)
    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    n, feat_dim1 = features1.shape
    m, feat_dim2 = features2.shape

    # Ensure that the feature dimensions match
    assert feat_dim1 == feat_dim2, "Feature dimensions must match"

    # Initialize the distance matrix
    dists = np.zeros((n, m))

    # Compute pairwise Euclidean distances
    for i in range(n):
        for j in range(m):
            # Calculate the Euclidean distance between feature i in features1
            # and feature j in features2
            dists[i, j] = np.linalg.norm(features1[i] - features2[j])

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).
    To start with, simply implement the "ratio test", equation 7.18 in
    section 7.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).
    You should call `compute_feature_distances()` in this function, and then
    process the output.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2
    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match
    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    # Compute feature distances
    dists = compute_feature_distances(features1, features2)

    # Initialize the result arrays
    matches = []
    confidences = []

    # Loop through the features in features1
    for i in range(len(features1)):
        # Sort the distances to the features in features2 for the current feature in features1
        sorted_indices = np.argsort(dists[i])

        # Calculate the ratio of the two smallest distances
        ratio = dists[i][sorted_indices[0]] / dists[i][sorted_indices[1]]

        # Apply the ratio test (typically, a ratio < 0.8 is used)
        if ratio < 0.8:
            matches.append([i, sorted_indices[0]])
            confidences.append(1.0 - ratio)

    # Convert the result lists to numpy arrays
    matches = np.array(matches)
    confidences = np.array(confidences)

    return matches, confidences

def pca(fvs1, fvs2, n_components= 24):
    """
    Perform PCA to reduce the number of dimensions in each feature vector resulting in a speed up.
    You will want to perform PCA on all the data together to obtain the same principle components.
    You will then resplit the data back into image1 and image2 features.

    Helpful functions: np.linalg.svd, np.mean, np.cov

    Args:
    -   fvs1: numpy nd-array of feature vectors with shape (k,128) for number of interest points 
        and feature vector dimension of image1
    -   fvs1: numpy nd-array of feature vectors with shape (m,128) for number of interest points 
        and feature vector dimension of image2
    -   n_components: m desired dimension of feature vector

    Return:
    -   reduced_fvs1: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image1
    -   reduced_fvs2: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image2
    """

    # Combine the feature vectors from both sets for PCA
    combined_features = np.vstack((fvs1, fvs2))

    # Calculate the mean vector
    mean_vector = np.mean(combined_features, axis=0)

    # Center the data by subtracting the mean
    centered_features = combined_features - mean_vector

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_features, rowvar=False)

    # Perform SVD to obtain the principal components
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Choose the top n_components principal components
    U_reduced = U[:, :n_components]

    # Project the centered feature vectors onto the reduced principal components
    reduced_features = np.dot(centered_features, U_reduced)

    # Split the reduced features back into image1 and image2 sets
    reduced_fvs1 = reduced_features[:len(fvs1)]
    reduced_fvs2 = reduced_features[len(fvs1):]

    return reduced_fvs1, reduced_fvs2

def accelerated_matching(features1, features2, x1, y1, x2, y2):
    """
    This method should operate in the same way as the match_features function you already coded.
    Try to make any improvements to the matching algorithm that would speed it up.
    One suggestion is to use a space partitioning data structure like a kd-tree or some
    third party approximate nearest neighbor package to accelerate matching.
    Note that doing PCA here does not count. This implementation MUST be faster than PCA
    to get credit.
    """

    # Build a KD-tree on features2
    kd_tree = KDTree(features2)

    # Initialize the result arrays
    matches = []
    confidences = []

    # Set a threshold for matching
    threshold = 0.8  # Adjust as needed

    # Loop through the features in features1
    for i in range(len(features1)):
        # Perform a k-Nearest Neighbors search using the KD-tree
        dists, indices = kd_tree.query([features1[i]], k=2)

        # Calculate the ratio of the two smallest distances
        ratio = dists[0][0] / dists[0][1]

        # Apply the ratio test
        if ratio < threshold:
            matches.append([i, indices[0][0]])
            confidences.append(1.0 - ratio)

    # Convert the result lists to numpy arrays
    matches = np.array(matches)
    confidences = np.array(confidences)

    return matches, confidences