import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from HoughCircleTransform import HoughCircleTransform

# # ==============  task 1.1 ===============
# # read images and convert them into grayscale
# # define images paths
# image_folder = "images"
# image_names = ["coins.png", "smarties.png"]
#
# # for coins.png
# image_coins_path = os.path.join(image_folder, image_names[0])  # image path
# image_coins_bgr = cv2.imread(image_coins_path)  # read image
# image_coins_gray = cv2.cvtColor(image_coins_bgr, cv2.COLOR_BGR2GRAY)  # convert to grayscale
#
# # for smarties.png
# image_smarties_path = os.path.join(image_folder, image_names[1])  # image path
# image_smarties_bgr = cv2.imread(image_smarties_path)  # read image
# image_smarties_gray = cv2.cvtColor(image_smarties_bgr, cv2.COLOR_BGR2GRAY)  # convert to grayscale
#
#
# # ==============  task 1.2 ==============
# def show_histogram_and_canny_result(image_bgr, image_gray, low_threshold, high_threshold):
#     """
#     # ========================  Task 1.2 ===========================
#     1. Show the gradient histogram of given image for better threshold choice
#     2. Apply CV2 canny edge detection with given parameters and show the result
#     """
#     # histogram
#     grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)  # compute Sobel gradient
#     grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)  # compute Sobel gradient
#     magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)  # compute gradient
#     # show histogram
#     plt.figure(figsize=(10, 5))
#     plt.hist(magnitude.ravel(), bins=256, range=[0, 256], color='black', histtype='step')
#     plt.title('Gradient Magnitude Histogram')
#     plt.xlabel('Gradient magnitude')
#     plt.ylabel('Pixel count')
#     plt.show()
#     # apply canny edge detection
#     image_edges = cv2.Canny(image_gray, low_threshold, high_threshold)  # use canny edge detect of cv2
#     # show the original image and the detection result
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121), plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
#     plt.subplot(122), plt.imshow(image_edges, cmap='gray'), plt.title('Edge Image')
#     plt.show()
#
#
# # ==============  task 1.2 ===============
# image_coins_edges = cv2.Canny(image_coins_gray, 20, 40)  # use canny edge detect of cv2
# image_smarties_edges = cv2.Canny(image_smarties_gray, 30, 65)  # use canny edge detect of cv2
# # show the detection result
# show_histogram_and_canny_result(image_coins_bgr, image_coins_gray, 20, 40)
# show_histogram_and_canny_result(image_smarties_bgr, image_smarties_gray, 30, 65)
#
#
# # ========================  Task 1.3 ===========================
# def hough_circle_transform(edges, min_radius, max_radius, radius_step, theta_step, threshold_ratio):
#     """
#     ========================  Task 1.3 ===========================
#     Performs Hough Circle Transform on the input edge image.
#
#     Args:
#         edges: Edge image.
#         min_radius: Minimum radius of the circles to be detected.
#         max_radius: Maximum radius of the circles to be detected.
#         radius_step: Increment step for the radius.
#         theta_step: Increment step for the angle theta (in degrees).
#         threshold_ratio: Ratio of the minimum score required to be considered a circle.
#
#     Returns:
#         A list of detected circles represented as [a, b, r],
#         where (a, b) is the center and r is the radius.
#     """
#     print("Executing hough circle transform detection with threshold ratio = " + str(threshold_ratio))
#     # Get the dimensions of the input image
#     rows, cols = edges.shape[:2]
#
#     # Initialize the accumulator array
#     accumulator = np.zeros((rows, cols, (max_radius - min_radius) // radius_step + 1))
#
#     # Get the indices of edge pixels
#     edge_pixels = np.argwhere(edges != 0)
#
#     # Iterate over the edge pixels
#     for ep in tqdm(edge_pixels):
#         x, y = ep
#
#         # Iterate over the possible radii
#         for r in range(min_radius, max_radius + 1, radius_step):
#             # Iterate over the possible theta values
#             for theta in range(0, 360, theta_step):
#                 theta_rad = np.deg2rad(theta)
#
#                 # Calculate the center coordinates
#                 a = int(x - r * np.cos(theta_rad))
#                 b = int(y - r * np.sin(theta_rad))
#
#                 # Increment the accumulator if the center is within the image bounds
#                 if 0 <= a < rows and 0 <= b < cols:
#                     accumulator[a, b, (r - min_radius) // radius_step] += 1
#
#     # Find the local maxima in the accumulator
#     max_score = np.max(accumulator)
#     threshold = max_score * threshold_ratio
#     circles = []
#     for r in range(accumulator.shape[2]):
#         acc_slice = accumulator[:, :, r]
#         circle_indices = np.argwhere(acc_slice >= threshold)
#         for idx in circle_indices:
#             a, b = idx
#             circles.append([a, b, min_radius + r * radius_step])
#
#     # Sort the circles based on the number of votes (accumulator values)
#     circles = sorted(circles, key=lambda x: accumulator[x[0], x[1], (x[2] - min_radius) // radius_step], reverse=True)
#     # Print the maximum score and the number of circles to be returned
#     print(f"Maximum score: {max_score}")
#     print(f"Number of circles to be returned: {len(circles)}")
#     return circles
#
#
# def show_hough_circle_transform_result(image_bgr, image_edges, threshold_ratio):
#     """
#     ========================  Task 1.3 ===========================
#     Draw detected circles on image and show
#     """
#     # get the circles detected by hough circle transformation
#     circles = hough_circle_transform(image_edges, min_radius=10, max_radius=100, radius_step=5, theta_step=10, threshold_ratio=threshold_ratio)
#     # draw circles on the original images
#     for circle in circles:
#         a, b, r = circle
#         cv2.circle(image_bgr, (b, a), r, (0, 255, 0), 1)  # green color
#     # Display the result
#     cv2.imshow('Detected Circles', image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # ========================  Task 1.3 ===========================
# # show the circles on images
# show_hough_circle_transform_result(image_coins_bgr, image_coins_edges, 0.6)
# show_hough_circle_transform_result(image_smarties_bgr, image_smarties_edges, 0.6)
#
#
# # ========================  Task 1.4 ===========================
# def non_max_suppression(circles, threshold):
#

"""
All function for Task 1 is implemented in HoughCircleTransform.py.
This file is only for executing the functions.
"""
# ========================  Task 1.1 ===========================
# define images paths, using os here
image_folder = "images"
image_names = ["coins.png", "smarties.png"]
image_coins_path = os.path.join(image_folder, image_names[0])  # image path
image_smarties_path = os.path.join(image_folder, image_names[1])  # image path
# instantiate HoughCircleTransform class
hough4coins = HoughCircleTransform(image_coins_path)
hough4smarties = HoughCircleTransform(image_smarties_path)

# ========================  Task 1.2 ===========================
coins_edges = hough4coins.canny_edge_detect(100,200)
smarties_edges = hough4smarties.canny_edge_detect(100,200)

# ========================  Task 1.3 ===========================
# get edges and apply hough transform then show result
# coins_hough = hough4coins.hough(coins_edges, 10, 100, 5, 10, 0.6)
# smarties_hough = hough4smarties.hough(smarties_edges, 10, 100, 5, 10, 0.6)

# ========================  Task 1.4 ===========================
coins_hough_nms = hough4coins.nms_hough(coins_edges, 10, 100, 5, 10, 0.6, 5)
smarties_hough_nms = hough4smarties.nms_hough(smarties_edges, 10, 100, 5, 10, 0.6, 5)
