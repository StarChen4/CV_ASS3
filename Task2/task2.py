import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Homography import homography_w_normalisation, match_and_save, homography_w_normalisation_ransac, warp_and_stitch

# ============================================ task 2.1 ====================================================
# set image path
image_folder = "images"
image1_name = "mountain1.jpg"
image2_name = "mountain2.jpg"
img1_path = os.path.join(image_folder, image1_name)
img2_path = os.path.join(image_folder, image2_name)
# read image using cv2
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# ============================================ task 2.4 ====================================================
# Detect SIFT keypoints, perform feature matching, and save the coordinates of matched points
match_and_save(img1, img2, 'npyfiles/pts1.npy', 'npyfiles/pts2.npy')
# ============================================ task 2.5 ====================================================
# Load the coordinates of matched points from .npy files
points1 = np.load('npyfiles/pts1.npy')
points2 = np.load('npyfiles/pts2.npy')
# Extract the u,v coordinates of the matched points
u1, v1 = points1[:, 0], points1[:, 1]
u2, v2 = points2[:, 0], points2[:, 1]
# Compute the homography matrix using the RANSAC algorithm
H_ransac = homography_w_normalisation_ransac(u1, v1, u2, v2, 1, 2000)
# ============================================ task 2.6 ====================================================
# Warp and stitch the images
result = warp_and_stitch(img1, img2, H_ransac)
# Display the stitched image
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save image after stitch
cv2.imwrite("output/stitched_image.jpg", result)
# ============================================ task 2.7 ====================================================
# BGR 2 RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# show images
plt.figure()
plt.subplot(121), plt.imshow(img1_rgb), plt.title('Image 1')
plt.subplot(122), plt.imshow(img2_rgb), plt.title('Image 2')
plt.show()
# Select points from Image 1
plt.imshow(img1_rgb)
points1 = plt.ginput(n=-1, timeout=0)  # Select points, press Enter to finish
plt.close()
# Select points from Image 2
plt.imshow(img2_rgb)
points2 = plt.ginput(n=-1, timeout=0)
plt.close()

if len(points1) >= 4 and len(points2) >= 4:
    points1 = np.array(points1)
    points2 = np.array(points2)

    # Save points to file
    np.save('npyfiles/manual_points1.npy', points1)
    np.save('npyfiles/manual_points2.npy', points2)
else:
    print("At least 4 pairs of points are required.")
u1, v1 = points1[:, 0], points1[:, 1]
u2, v2 = points2[:, 0], points2[:, 1]
H_ground = homography_w_normalisation_ransac(u1, v1, u2, v2)
print(f"Gound truth by manually select points : \n {H_ground}")
# Warp and stitch the images
result = warp_and_stitch(img1, img2, H_ground)
# Display the stitched image
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save image after stitch
cv2.imwrite("output/stitched_image_with_manual_points.jpg", result)


