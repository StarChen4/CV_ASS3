import cv2
import os

from Task1.HoughCircleTransform import hough_circle_transform, draw_circles, non_maximum_suppression, show_image, hough_circle_transform_outside_image


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
# read images and convert to grayscale
# for coins.png
image_coins_bgr = cv2.imread(image_coins_path)  # read image
image_coins_gray = cv2.cvtColor(image_coins_bgr, cv2.COLOR_BGR2GRAY)  # convert to grayscale
# for smarties.png
image_smarties_bgr = cv2.imread(image_smarties_path)  # read image
image_smarties_gray = cv2.cvtColor(image_smarties_bgr, cv2.COLOR_BGR2GRAY)  # convert to grayscale

# ========================  Task 1.2 ===========================
# apply canny edge detection on images
image_coins_edges = cv2.Canny(image_coins_gray, 108, 211)  # use canny edge detect of cv2
image_smarties_edges = cv2.Canny(image_smarties_gray, 30, 65)  # use canny edge detect of cv2

# ========================  Task 1.3 ===========================
# apply hough circle transformation on images to get circles
print("Hough circle transformation: processing coins.png")
coins_circles = hough_circle_transform(image_coins_edges, 20, 50, 2, 10, 0.7)  # 20,50,2,10,0.7
print("Hough circle transformation: processing smarties.png")
smarties_circles = hough_circle_transform(image_smarties_edges, 20, 50, 2, 10, 0.6)  # 20,50,2,10,0.6
# draw circles on original images
image_coins_hough = draw_circles(image_coins_bgr, coins_circles)
image_smarties_hough = draw_circles(image_smarties_bgr, smarties_circles)
# save result images as [image_name]_hough_result.jpg
cv2.imwrite("results/coins_hough_result.jpg", image_coins_hough)
cv2.imwrite("results/smarties_hough_result.jpg", image_smarties_hough)
# plot images after hough
show_image(image_coins_bgr, image_coins_hough, "Hough Circle Transform")
show_image(image_smarties_bgr, image_smarties_hough, "Hough Circle Transform")

# ========================  Task 1.4 ===========================
# apply nms on hough result
print("NMS: processing coins.png")
coins_circles_nms = non_maximum_suppression(coins_circles, 20)
print("NMS: processing smarties.png")
smarties_circles_nms = non_maximum_suppression(smarties_circles, 10)
# draw nms hough circles on original images
image_coins_hough_nms = draw_circles(image_coins_bgr, coins_circles_nms)
image_smarties_hough_nms = draw_circles(image_smarties_bgr, smarties_circles_nms)
# save result images as [image_name]_hough_nms_result.jpg
cv2.imwrite("results/coins_hough_nms_result.jpg", image_coins_hough_nms)
cv2.imwrite("results/smarties_hough_nms_result.jpg", image_smarties_hough_nms)
# plot images after nms
show_image(image_coins_bgr, image_coins_hough_nms, "Hough Circle Transform with NMS")
show_image(image_smarties_bgr, image_smarties_hough_nms, "Hough Circle Transform with NMS")

# # ========================  Task 1.6 ===========================
# # detect circles even if they are outside the image range
# print("Hough circle transformation: processing coins.png")
# coins_circles_contain_outside = hough_circle_transform_outside_image(image_coins_edges, 25, 50, 2, 10, 0.7)
# print("Hough circle transformation: processing smarties.png")
# smarties_circles_contain_outside = hough_circle_transform_outside_image(image_smarties_edges, 25, 50, 2, 10, 0.6)
# print("NMS: processing coins.png")
# coins_circles_nms_outside = non_maximum_suppression(coins_circles_contain_outside, 10)
# print("NMS: processing smarties.png")
# smarties_circles_nms_outside = non_maximum_suppression(smarties_circles_contain_outside, 10)
# # draw nms hough circles on original images
# image_coins_hough_nms_outside = draw_circles(image_coins_bgr, coins_circles_nms_outside)
# image_smarties_hough_nms_outside = draw_circles(image_smarties_bgr, smarties_circles_nms_outside)
# # save result images as [image_name]_hough_nms_outside_result.jpg
# cv2.imwrite("results/coins_hough_nms_outside_result.jpg", image_coins_hough_nms_outside)
# cv2.imwrite("results/smarties_hough_nms_outside_result.jpg", image_smarties_hough_nms_outside)
# # plot images after nms
# show_image(image_coins_bgr, image_coins_hough_nms_outside, "Hough Circle Transform with NMS")
# show_image(image_smarties_bgr, image_smarties_hough_nms_outside, "Hough Circle Transform with NMS")








