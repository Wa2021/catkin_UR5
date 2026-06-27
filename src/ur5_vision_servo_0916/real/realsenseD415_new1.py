# RealSense D415 相机类：后台线程持续取帧，get_data() 返回最新缓存帧。

import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import time

class Camera(object):

    def __init__(self, width=640, height=480, fps=30):
        self.im_height = height
        self.im_width = width
        self.fps = fps

        # 后台线程和最新帧缓存
        self.pipeline = None
        self.scale = None
        self.intrinsics = None

        self.align = None

        self.latest_color_frame = None
        self.latest_depth_frame = None

        self.thread = None
        self.lock = threading.Lock()
        self.running = False

        # 在初始化时就直接启动相机和线程
        self.start()

    def start(self):
        """
        启动相机硬件并开始后台线程，取代了旧的 connect() 方法。
        """
        if self.running:
            print("相机已经在运行中。")
            return

        print("正在启动 RealSense 相机...")
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # Start streaming
        try:
            cfg = self.pipeline.start(config)

            # Determine intrinsics
            rgb_profile = cfg.get_stream(rs.stream.color)
            self.intrinsics = self.get_intrinsics(rgb_profile)

            # Determine depth scale
            self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

            # Align depth frames to the color stream once and reuse the object.
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            print("相机深度缩放因子 (depth scale):", self.scale)
            print("D415/D435 已连接...")
        except Exception as e:
            print(f"错误: 无法启动相机。{e}")
            return

        # 启动后台线程
        self.running = True
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()
        print("相机数据流后台线程已启动。")
        # 等待一下，让线程有时间获取第一帧
        time.sleep(1.0)

    def _update_frames(self):
        """
        线程的主体函数，在后台不停地运行，
        负责从相机硬件获取帧并更新内部的 latest_..._frame 变量。
        """
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                aligned_frames = self.align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                # --- 将原始帧数据转换为Numpy数组 ---
                # 深度图像
                depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self.scale
                depth_image = np.expand_dims(depth_image, axis=2)  # 保持与你原来代码一致的维度

                # 彩色图像
                color_image = np.asanyarray(color_frame.get_data())

                # 使用线程锁来安全地更新最新的帧
                with self.lock:
                    self.latest_color_frame = color_image
                    self.latest_depth_frame = depth_image

            except Exception as e:
                print(f"相机线程错误: {e}")
                time.sleep(0.5)

    def get_data(self):
        """
        从后台线程准备好的最新帧中获取图像，避免主流程直接阻塞在硬件读取上。
        """
        if not self.running:
            print("错误：相机未运行，请先调用 start()。")
            return None, None

        if self.latest_color_frame is None or self.latest_depth_frame is None:
            return None, None

        # 使用线程锁来安全地读取最新的帧
        with self.lock:
            return self.latest_color_frame, self.latest_depth_frame

    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(1)

    def get_intrinsics(self,rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        return intrinsics

    def stop(self):
        """
        安全地停止相机和后台线程。
        """
        if not self.running:
            return

        print("正在停止相机...")
        self.running = False
        if self.thread is not None:
            self.thread.join()  # 等待线程完全结束
        if self.pipeline is not None:
            self.pipeline.stop()
        print("相机已安全停止。")


if __name__== '__main__':
    mycamera = Camera()
    # mycamera.get_data()
    mycamera.plot_image()
    # print(mycamera.intrinsics)
