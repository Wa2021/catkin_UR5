#需要开启机械臂
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont  # 添加PIL库支持中文
from UR_Robot import UR_Robot

# 加载标定结果
with open("../calibration_result1.2.1.json", "r") as f:
    data = json.load(f)
camera_matrix = np.array(data["camera_matrix"])
dist_coeffs = np.array(data["dist_coeffs"])
# 初始化机器人和相机
robot = UR_Robot("192.168.0.1")
rgb_img, _ = robot.get_camera_data()
# 检测棋盘格
pattern_size = (8, 5)  # 你的角点模式
square_size = 0.027  # 每个格子的边长（米）
gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    # 使用PIL添加中文文本
    img_pil = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 使用支持中文的字体（确保系统有中文字体）
    try:
        font = ImageFont.truetype("simhei.ttf", 30)  # 黑体
    except:
        font = ImageFont.load_default()

    draw.text((50, 50), "❌ 未检测到棋盘格，请调整姿态", font=font, fill=(0, 255, 0))

    # 转换回OpenCV格式
    result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow(" 错误提示", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
# 精细化角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
# 准备 object_points（棋盘格世界坐标）
pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
pattern_points *= square_size
# 使用 cv2.solvePnP  拟合 rvec, tvec
ret, rvec, tvec = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coeffs)
# 投影棋盘格点
projected_points, _ = cv2.projectPoints(pattern_points, rvec, tvec, camera_matrix, dist_coeffs)
# 画点对比：实际角点（绿） vs 投影角点（红）
for pt1, pt2 in zip(corners, projected_points):
    pt_actual = tuple(pt1.ravel().astype(int))
    pt_proj = tuple(pt2.ravel().astype(int))
    cv2.circle(rgb_img, pt_actual, 4, (0, 255, 0), 1)  # 绿点：实际检测
    cv2.circle(rgb_img, pt_proj, 2, (0, 0, 255), -1)  # 红点：投影位置
# 使用PIL添加中文标题
img_pil = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
try:
    # 尝试使用黑体（Windows系统自带）
    font = ImageFont.truetype("simhei.ttf", 30)
    # 或者使用微软雅黑
    # font = ImageFont.truetype("msyh.ttc",  30)
except:
    # 回退到默认字体（可能不支持中文）
    font = ImageFont.load_default()
# 添加标题
title = "内参验证：绿=实际角点，红=投影点"
draw.text((20, 20), title, font=font, fill=(255, 255, 255))
# 添加图例说明
legend = "绿圈: 检测角点 | 红点: 投影位置"
draw.text((20, img_pil.height - 40), legend, font=font, fill=(255, 255, 255))
# 转换回OpenCV格式
result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# 显示结果
cv2.imshow("Calibration  Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()