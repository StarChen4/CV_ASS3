import cv2
import numpy as np
import matplotlib.pyplot as plt


def homography(u1, v1, u2, v2):
    """                         Task 2.2
    Computes the homography H using the Direct Linear Transformation
    Arguments:
    u1, v1: normalised (u,v) coordinates from image 1
    u2, v2: normalised (u,v) coordinates from image 2
    Output:
    H: the 3×3 homography matrix that warps normalised coordinates
    from image 1 into normalised coordinates from image 2
    """
    # Assert that the input coordinates have the same length
    assert len(u1) == len(v1) == len(u2) == len(v2), "Input coordinates must have the same length"
    # get the number of points, at least 4
    num_points = len(u1)
    if num_points < 4:
        raise ValueError("At least 4 points are needed")
    A = []   # initialize A
    for i in range(num_points):  # in each pair of points
        x1, y1 = u1[i], v1[i]
        x2, y2 = u2[i], v2[i]
        # Construct the coefficient matrix A of the linear system
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    # Convert A to a numpy array
    A = np.array(A)
    # Perform SVD decomposition on A
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]  # The homography matrix H is the last row of Vh
    H = L.reshape(3, 3)  # reshape to 3×3
    return H


def compute_normalisation_matrix(points):
    """                         Task 2.3
    Computes the normalization transformation matrix for homogeneous point coordinates.
    Arguments:
    points: Array of points with shape (N, 2) where N is the number of points.
    Output:
    T: The normalization transformation matrix.
    """
    mean = np.mean(points, axis=0)  # Compute the mean of the points
    std_dev = np.std(points, axis=0)  # Compute the standard deviation of the points
    # Compute the normalization scale factor
    scale = np.sqrt(2) / std_dev
    # Construct the normalization transformation matrix T
    T = np.array([[scale[0], 0, -scale[0] * mean[0]],
                  [0, scale[1], -scale[1] * mean[1]],
                  [0, 0, 1]])
    return T


def homography_w_normalisation(u1, v1, u2, v2):
    """                         Task 2.3
    Computes the homography matrix with normalization using the DLT algorithm.
    Arguments:
    u1, v1: normalised (u,v) coordinates from image 1
    u2, v2: normalised (u,v) coordinates from image 2
    Output:
    H: the 3×3 homography matrix that warps normalised coordinates from image 1 into normalised coordinates from image 2
    """
    # Stack the input coordinates into point arrays
    points1 = np.stack((u1, v1), axis=-1)
    points2 = np.stack((u2, v2), axis=-1)
    # Compute the normalization transformation matrices T1 and T2
    T1 = compute_normalisation_matrix(points1)
    T2 = compute_normalisation_matrix(points2)
    # Normalize the point coordinates
    ones = np.ones((points1.shape[0], 1))
    normalized_points1 = (T1 @ np.hstack((points1, ones)).T).T
    normalized_points2 = (T2 @ np.hstack((points2, ones)).T).T
    # Extract the normalized point coordinates
    u1_norm = normalized_points1[:, 0]
    v1_norm = normalized_points1[:, 1]
    u2_norm = normalized_points2[:, 0]
    v2_norm = normalized_points2[:, 1]
    # Compute the homography matrix H_norm using the normalized coordinates
    H_norm = homography(u1_norm, v1_norm, u2_norm, v2_norm)
    # De-normalize the homography matrix
    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]  # Normalize the homography matrix so that its last element is 1

    return H


def match_and_save(img1, img2, save_path1, save_path2):
    #                             Task 2.4
    # Detects SIFT keypoints, performs feature matching, and saves the coordinates of matched points.
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Initialize the SIFT feature detector
    sift = cv2.SIFT_create()
    # Detect SIFT keypoints and compute descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    # Sort the matches based on feature distance
    matches = sorted(matches, key=lambda x: x.distance)
    np.random.seed(42)  # Set a fixed random seed for reproducibility
    # Randomly select half of the matches
    num_matches = len(matches)
    selected_matches = np.random.choice(matches, size=num_matches // 2, replace=False)
    # Extract the coordinates of the matched points
    points1 = np.zeros((len(selected_matches), 2))
    points2 = np.zeros((len(selected_matches), 2))
    for i, match in enumerate(selected_matches):
        points1[i, :] = keypoints_1[match.queryIdx].pt
        points2[i, :] = keypoints_2[match.trainIdx].pt
    # Save the coordinates of the matched points as .npy files
    np.save(save_path1, points1)
    np.save(save_path2, points2)


def homography_w_normalisation_ransac(u1, v1, u2, v2, threshold=1.0, max_iterations=1000):
    """                             Task 2.5
        Computes the homography matrix using the RANSAC algorithm and normalized DLT algorithm
        Arguments:
        u1, v1: (u,v) coordinates from image 1
        u2, v2: (u,v) coordinates from image 2
        threshold: Distance threshold for the RANSAC algorithm, default is 1.0
        max_iterations: Maximum number of iterations for the RANSAC algorithm, default is 1000
        Output:
        best_H: The best homography matrix computed using the RANSAC algorithm and normalized DLT algorithm
        """
    num_points = len(u1)
    best_H = None
    max_inliers = 0

    for _ in range(max_iterations):
        # Randomly select 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        u1_sample = u1[indices]
        v1_sample = v1[indices]
        u2_sample = u2[indices]
        v2_sample = v2[indices]
        # Compute the homography matrix H using the selected 4 points
        H = homography_w_normalisation(u1_sample, v1_sample, u2_sample, v2_sample)
        # Stack the point coordinates into arrays
        ones = np.ones((num_points, 1))
        points1 = np.stack((u1, v1), axis=-1)
        points2 = np.stack((u2, v2), axis=-1)
        # Apply the homography matrix H to transform the points
        projected_points = H @ np.vstack((points1.T, np.ones(num_points)))
        projected_points /= projected_points[2, :]
        projected_points = projected_points[:2, :].T
        # Compute the errors between the projected points and the actual points
        errors = np.sqrt(np.sum((points2 - projected_points) ** 2, axis=1))
        # Determine the inliers (points with errors below the threshold)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)
        # Update the best homography matrix and the maximum number of inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
    print(f'Max inliers number {max_inliers}')
    print(f'Result H \n {H}')
    return best_H


def warp_and_stitch(img1, img2, H):
    """                             Task 2.6
        Warps and stitches the images using the homography matrix
        Arguments:
        img1: Image 1
        img2: Image 2
        H: Homography matrix
        Output:
        result_img: The resulting image after warping and stitching
        """
    # Get the height and width of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # Define the four corner coordinates of image 1
    corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype='float32')
    # Transform the corner points of image 1 using the homography matrix H
    corners_transformed_img1 = cv2.perspectiveTransform(np.array([corners_img1]), H)[0]
    # Define the four corner coordinates of image 2
    corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype='float32')
    # Combine the transformed corner points of image 1 and the corner points of image 2
    all_corners = np.vstack((corners_transformed_img1, corners_img2))
    # Compute the minimum and maximum coordinate values of the combined corner points
    [xmin, ymin] = np.int32(all_corners.min(axis=0) - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0) + 0.5)
    # Compute the translation amounts
    t = [-xmin, -ymin]
    H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # Construct the translation matrix
    # Apply perspective transformation and translation to image 1
    result_img = cv2.warpPerspective(img1, H_translation @ H, (xmax - xmin, ymax - ymin))
    # Copy image 2 to the corresponding position in the resulting image
    result_img[t[1]:h2 + t[1], t[0]:w2 + t[0]] = img2

    return result_img



