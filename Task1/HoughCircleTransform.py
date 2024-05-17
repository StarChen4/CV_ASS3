import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def draw_circles(image, circles):
    """
    Draw circles on an image copy with green color
    """
    # Convert BGR image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert grayscale image back to BGR to draw colored circles
    output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for circle in circles:
        a, b, r = circle
        cv2.circle(output_image, (b, a), r, (0, 255, 0), 2)
    return output_image


def show_image(img_bgr, img_compare, title):
    # using plt so show the image before and after processing
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.subplot(121), plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB)), plt.title('Edge Image')
    plt.show()


def hough_circle_transform(edges, min_radius, max_radius, radius_step, angle_step, threshold_ratio):
    """
    Task1.3
    Apply hough circle transformation on an edge image.
    """
    # Get the dimensions of the input image
    height, width = edges.shape

    # Initialize the accumulator array
    num_radii = (max_radius - min_radius) // radius_step + 1  # Calculate number of radius
    accumulator = np.zeros((height, width, num_radii), dtype=np.uint64)

    # Get the indices of edge pixels
    edge_pixels = np.argwhere(edges != 0)

    # precompute angles and cos, sin
    angles = np.arange(0, 360, angle_step)
    cos_theta = np.cos(np.deg2rad(angles))
    sin_theta = np.sin(np.deg2rad(angles))

    # Iterate over all edge pixels and radius
    for x, y in tqdm(edge_pixels, desc="Processing edge pixels", total=len(edge_pixels)):
        for radius_index, radius in enumerate(range(min_radius, max_radius, radius_step)):
            for cos_t, sin_t in zip(cos_theta, sin_theta):
                a = int(x - radius * cos_t)  # Calculating a using formula a = x − r cos θ
                b = int(y - radius * sin_t)  # Calculating b using formula b = y − r sin θ
                # If within this image range, add the score by 1
                if 0 <= a < height and 0 <= b < width:
                    accumulator[a, b, radius_index] += 1

    # Find the maxima in the accumulator using threshold ratio
    max_acc = np.max(accumulator)
    threshold = threshold_ratio * max_acc
    circles = []  # List to save the result

    # Apply thresholding
    for r in range(accumulator.shape[2]):
        acc_slice = accumulator[:, :, r]  # get a slice of the accumulator
        circle_indices = np.argwhere(acc_slice >= threshold)  # get the indices of circles with score is over threshold
        for idx in circle_indices:
            a, b = idx
            circles.append([a, b, min_radius + r * radius_step])  # append the circle into result List

    # Sort by descending order of the vote score of each circle
    circles = sorted(circles, key=lambda x: accumulator[x[0], x[1], (x[2] - min_radius) // radius_step], reverse=True)

    return circles


def non_maximum_suppression(circles, threshold_pixels):
    """
    Task 1.4
    Apply nms on hough circle detection result
    """
    # To save result circles
    suppressed_circles = []

    # A marker to mark circles that are suppressed
    suppressed = np.zeros(len(circles), dtype=bool)

    for i, (a1, b1, r1) in tqdm(enumerate(circles)):
        # skip suppressed circles
        if suppressed[i]:
            continue

        for j, (a2, b2, r2) in enumerate(circles[i+1:], start=i+1):  # Other circles
            # skip suppressed circles
            if suppressed[j]:
                continue

            # Compare the difference of a, b and r. If below threshold, suppress it.
            if abs(a1 - a2) <= threshold_pixels and abs(b1 - b2) <= threshold_pixels and abs(r1 - r2) <= threshold_pixels:
                suppressed[j] = True

        if not suppressed[i]:
            suppressed_circles.append((a1, b1, r1))

    return suppressed_circles


def hough_circle_transform_outside_image(edges, min_radius, max_radius, radius_step, angle_step, threshold_ratio):
    """
    Task1.3
    Apply hough circle transformation on an edge image.
    """
    # Get the dimensions of the input image
    height, width = edges.shape

    # Initialize the accumulator array
    max_offset = max_radius  #
    acc_height = height + 2 * max_offset
    acc_width = width + 2 * max_offset
    num_radii = (max_radius - min_radius) // radius_step + 1  # Calculate number of radius
    accumulator = np.zeros((acc_height, acc_width, num_radii), dtype=np.uint64)

    # Get the indices of edge pixels
    edge_pixels = np.argwhere(edges != 0)

    # precompute angles and cos, sin
    angles = np.arange(0, 360, angle_step)
    cos_theta = np.cos(np.deg2rad(angles))
    sin_theta = np.sin(np.deg2rad(angles))

    # Iterate over all edge pixels and radius
    for x, y in tqdm(edge_pixels, desc="Processing edge pixels", total=len(edge_pixels)):
        for radius_index, radius in enumerate(range(min_radius, max_radius, radius_step)):
            for cos_t, sin_t in zip(cos_theta, sin_theta):
                a = int(x - radius * cos_t)  # Calculating a using formula a = x − r cos θ
                b = int(y - radius * sin_t)  # Calculating b using formula b = y − r sin θ
                # If within the extended accumulator array
                if 0 <= a < acc_height and 0 <= b < acc_width:
                    accumulator[a, b, radius_index] += 1

    # Find the maxima in the accumulator using threshold ratio
    max_acc = np.max(accumulator)
    threshold = threshold_ratio * max_acc
    circles = []  # List to save the result

    # Apply thresholding
    for r in range(accumulator.shape[2]):
        acc_slice = accumulator[:, :, r]  # get a slice of the accumulator
        circle_indices = np.argwhere(acc_slice >= threshold)  # get the indices of circles with score is over threshold
        for idx in circle_indices:
            a, b = idx
            a -= max_offset  # adjust the offset to go back to original coordinates
            b -= max_offset  # adjust the offset to go back to original coordinates
            circles.append([a, b, min_radius + r * radius_step])  # append the circle into result List

    # Sort by descending order of the vote score of each circle
    circles = sorted(circles, key=lambda x: accumulator[x[0], x[1], (x[2] - min_radius) // radius_step], reverse=True)

    return circles




