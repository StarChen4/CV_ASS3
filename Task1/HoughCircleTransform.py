import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class HoughCircleTransform:
    def __init__(self, img_path):
        """
        ======================  task 1.1 =========================
        Use cv2 to read and convert to gray scale
        """
        self.img_path = img_path
        self.img_bgr = cv2.imread(self.img_path)
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

    def canny_edge_detect(self, canny_low, canny_high):
        """
        ======================  task 1.2 =========================
        Apply canny edge detection on image and show the result
        :param canny_low: Lower threshold of canny detection
        :param canny_high: Higher threshold of canny detection

        :return: a nparray image containing edges
        """
        # apply gaussian blur
        img_gray = cv2.GaussianBlur(self.img_gray, (3,3),0)
        img_edges = cv2.Canny(self.img_gray, canny_low, canny_high)
        # show the original image and the detection result
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        plt.subplot(122), plt.imshow(img_edges, cmap='gray'), plt.title('Edge Image')
        plt.show()
        return img_edges

    def hough(self, edges, min_radius, max_radius, radius_step, theta_step, threshold_ratio):
        """
        ======================  task 1.3 =========================
        Apply hough circle transform on image and show the result
        :param edges: edges img for transformation
        :param min_radius: Minimum radius of the circles to be detected
        :param max_radius: Maximum radius of the circles to be detected
        :param radius_step: Increment step for the radius
        :param theta_step: Increment step for the angle theta (in degrees)
        :param threshold_ratio: Ratio of the minimum score required to be considered a circle

        :return: A list of detected circles represented as [a, b, r],
        where (a, b) is the center and r is the radius.
        """
        print("Executing hough circle transform detection with threshold ratio = " + str(threshold_ratio))
        # Get the dimensions of the input image
        rows, cols = edges.shape[:2]

        # Initialize the accumulator array
        accumulator = np.zeros((rows, cols, (max_radius - min_radius) // radius_step + 1))

        # Get the indices of edge pixels
        edge_pixels = np.argwhere(edges != 0)

        # Iterate over the edge pixels
        for ep in tqdm(edge_pixels):
            x, y = ep

            # Iterate over the possible radii
            for r in range(min_radius, max_radius + 1, radius_step):
                # Iterate over the possible theta values
                for theta in range(0, 360, theta_step):
                    # convert degree into rad
                    theta_rad = np.deg2rad(theta)

                    # Calculate the center coordinates
                    a = int(x - r * np.cos(theta_rad))
                    b = int(y - r * np.sin(theta_rad))

                    # Increment the accumulator if the center is within the image bounds
                    if 0 <= a < rows and 0 <= b < cols:
                        accumulator[a, b, (r - min_radius) // radius_step] += 1

        # Find the local maxima in the accumulator
        max_score = np.max(accumulator)
        threshold = max_score * threshold_ratio
        circles = []
        for r in range(accumulator.shape[2]):
            acc_slice = accumulator[:, :, r]
            circle_indices = np.argwhere(acc_slice >= threshold)
            for idx in circle_indices:
                a, b = idx
                circles.append([a, b, min_radius + r * radius_step])

        # Sort the circles based on the number of votes (accumulator values)
        circles = sorted(circles, key=lambda x: accumulator[x[0], x[1], (x[2] - min_radius) // radius_step],
                         reverse=True)
        # Print the maximum score and the number of circles to be returned
        print(f"Maximum score: {max_score}")
        print(f"Number of circles to be returned: {len(circles)}")

        # draw circles on the original images
        img_bgr_copy = self.img_bgr.copy()  # get a copy of original image
        for circle in circles:
            a, b, r = circle
            cv2.circle(img_bgr_copy, (b, a), r, (0, 255, 0), 1)  # green color
        # Display the result
        cv2.imshow('Detected Circles', img_bgr_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return circles, accumulator

    def nms_hough(self, edges, min_radius, max_radius, radius_step, theta_step, threshold_ratio, nms_threshold):
        """
        ======================  task 1.4 =========================
        Apply non-maximum suppression on the result of hough transformation
        :param circles: List of detected circles, each represented as [a, b, r] where (a, b) is the center and r is the radius.
        :param threshold: Threshold value for suppressing similar circles.

        :return: A list of circles after non-maximum suppression.
        """
        circles, accumulator = self.hough(edges, min_radius, max_radius, radius_step, theta_step, threshold_ratio)
        print("Applying Non Maximum Suppression with threshold = " + str(nms_threshold))
        # If there are no circles, return an empty list
        if len(circles) == 0:
            return []

        # Convert circles to numpy array for easier indexing
        circles = np.array(circles)

        # Get the scores (accumulator values) for each circle
        scores = np.array([accumulator[int(a), int(b), int((r - min_radius) / radius_step)] for a, b, r in circles])

        # Sort the circles based on their scores in descending order
        order = np.argsort(scores)[::-1]
        circles = circles[order]
        scores = scores[order]

        # Initialize an array to keep track of suppressed circles
        suppressed = np.zeros(len(circles), dtype=bool)

        # Iterate over the circles
        for i in range(len(circles)):
            if suppressed[i]:
                continue

            # Compare the current circle with the remaining circles
            for j in range(i+1, len(circles)):
                if suppressed[j]:
                    continue

                # Calculate the difference in center coordinates and radii
                da = circles[i, 0] - circles[j, 0]
                db = circles[i, 1] - circles[j, 1]
                dr = circles[i, 2] - circles[j, 2]

                # If the differences are smaller than the threshold, suppress the circle with lower score
                if abs(da) < nms_threshold and abs(db) < nms_threshold and abs(dr) < nms_threshold:
                    suppressed[j] = True

        print(f"Number of circles after suppression: {len(circles[~suppressed])}")
        # Draw the result on original image
        img_bgr_copy = self.img_bgr.copy()  # get a copy of original image
        for circle in circles[~suppressed]:
            a, b, r = circle
            cv2.circle(img_bgr_copy, (b, a), r, (0, 255, 0), 1)  # green color
        # Display the result
        cv2.imshow('Detected Circles after NMS', img_bgr_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the circles that were not suppressed
        return circles[~suppressed]





