#只能在windows下使用
import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色和深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动设备
print("启动 RealSense 相机...")
pipeline.start(config)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 转为 NumPy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度图转为伪彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 拼接两个图像（左右并排显示）
        images = np.hstack((color_image, depth_colormap))

        # 显示图像窗口
        cv2.imshow("RealSense D415 - RGB + Depth", images)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 键退出
            break
        elif key == ord('s'):
            # 保存图像
            cv2.imwrite("color_image.png", color_image)
            cv2.imwrite("depth_colormap.png", depth_colormap)
            print("已保存 color_image.png 和 depth_colormap.png")

finally:
    # 释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
    print("相机关闭。")
