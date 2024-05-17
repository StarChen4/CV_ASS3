import cv2
import numpy as np

def estimate_radius(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化半径列表
    radii = []

    # 遍历所有轮廓，计算最小外接圆的半径
    for contour in contours:
        if len(contour) > 0:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radii.append(radius)

    return radii

# 估算硬币的半径
coins_radii = estimate_radius('images/coins.png')
print(f"硬币的半径估算值（像素）：{coins_radii}")

# 估算糖果的半径
smarties_radii = estimate_radius('images/smarties.png')
print(f"糖果的半径估算值（像素）：{smarties_radii}")
