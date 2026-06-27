# RealSense D435 相机类：后台线程持续取帧，get_data() 返回最新缓存帧。

import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import time

class Camera(object):
    """
    RealSense D435 相机类
    特点：
    1. 后台线程持续获取图像，避免 get_data() 阻塞
    2. 自动获取相机内参，无需手动配置
    3. 支持自定义分辨率和帧率
    4. 线程安全的数据访问
    
    与 D415 的主要区别：
    - D435 视场角更广（87°×58° vs 65°×40°）
    - D435 最小工作距离更近（0.2m vs 0.3m）
    - 内参由相机自动提供，更准确
    """

    def __init__(self, width=640, height=480, fps=30):
        """
        初始化 RealSense D435 相机
        
        参数:
            width: 图像宽度（默认640）
            height: 图像高度（默认480）
            fps: 帧率（默认30）
            
        常用分辨率组合:
            - 640×480 @ 30fps（推荐，适合实时处理）
            - 1280×720 @ 30fps（高清，计算量更大）
            - 848×480 @ 60fps（宽屏，高帧率）
        """
        self.im_height = height
        self.im_width = width
        self.fps = fps

        # --- 核心变量 ---
        self.pipeline = None
        self.scale = None          # 深度缩放因子
        self.intrinsics = None     # 相机内参矩阵 3×3
        self.align = None          # 对齐对象（深度对齐到彩色）

        # 最新的帧数据
        self.latest_color_frame = None
        self.latest_depth_frame = None

        # 线程控制
        self.thread = None
        self.lock = threading.Lock()
        self.running = False

        # 启动相机
        self.start()

    def start(self):
        """
        启动相机硬件并开始后台线程
        """
        if self.running:
            print("[D435] 相机已经在运行中。")
            return

        print(f"[D435] 正在启动 RealSense D435 相机 ({self.im_width}×{self.im_height} @ {self.fps}fps)...")
        
        # 配置深度和彩色流
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        try:
            # 启动相机流
            cfg = self.pipeline.start(config)

            # 获取相机内参（自动从硬件读取）
            rgb_profile = cfg.get_stream(rs.stream.color)
            self.intrinsics = self.get_intrinsics(rgb_profile)

            # 获取深度缩放因子
            self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

            # 创建对齐对象（将深度图对齐到彩色图）
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            print(f"[D435] 相机深度缩放因子 (depth scale): {self.scale}")
            print(f"[D435] 相机内参:\n{self.intrinsics}")
            print("[D435] ✅ RealSense D435 已连接")
            
        except Exception as e:
            print(f"[D435] ❌ 错误: 无法启动相机。{e}")
            return

        # 启动后台线程
        self.running = True
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()
        print("[D435] 相机数据流后台线程已启动。")
        
        # 等待线程获取第一帧
        time.sleep(1.0)

    def _update_frames(self):
        """
        后台线程主函数：持续从相机获取帧并更新最新数据
        """
        while self.running:
            try:
                # 等待新的一帧（超时1秒）
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                # 对齐深度图到彩色图
                aligned_frames = self.align.process(frames)

                # 获取对齐后的深度和彩色帧
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                # 转换为 numpy 数组
                # 深度图：uint16 原始值 * scale -> 米为单位的 float
                depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self.scale
                depth_image = np.expand_dims(depth_image, axis=2)  # [H, W] -> [H, W, 1]

                # 彩色图：BGR 格式
                color_image = np.asanyarray(color_frame.get_data())

                # 线程安全地更新最新帧
                with self.lock:
                    self.latest_color_frame = color_image
                    self.latest_depth_frame = depth_image

            except Exception as e:
                print(f"[D435] 相机线程错误: {e}")
                time.sleep(0.5)

    def get_data(self):
        """
        获取最新的彩色和深度图像（非阻塞，快速返回）
        
        返回:
            color_image: numpy数组，shape=(H,W,3)，BGR格式
            depth_image: numpy数组，shape=(H,W,1)，单位为米
            
        注意：
            如果相机未就绪或线程未获取到数据，返回 (None, None)
        """
        if not self.running:
            print("[D435] ❌ 错误：相机未运行，请先调用 start()。")
            return None, None

        if self.latest_color_frame is None or self.latest_depth_frame is None:
            print("[D435] ⚠️ 警告：相机数据尚未就绪，请稍候。")
            return None, None

        # 线程安全地读取最新帧
        with self.lock:
            return self.latest_color_frame.copy(), self.latest_depth_frame.copy()

    def plot_image(self):
        """
        可视化彩色图和深度图（用于测试和调试）
        """
        color_image, depth_image = self.get_data()
        
        if color_image is None or depth_image is None:
            print("[D435] 无法可视化：图像数据不可用")
            return

        # 将深度图转换为伪彩色（方便观察）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # 如果尺寸不匹配，调整彩色图大小
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                color_image, 
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA
            )
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # 显示图像
        cv2.namedWindow('RealSense D435', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense D435', images)
        cv2.waitKey(1)

    def get_intrinsics(self, rgb_profile):
        """
        获取相机内参矩阵
        
        参数:
            rgb_profile: RealSense 彩色流配置
            
        返回:
            intrinsics: 3×3 numpy数组
            [[fx,  0, ppx],
             [ 0, fy, ppy],
             [ 0,  0,   1]]
        """
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print(f"[D435] 相机原始内参: {raw_intrinsics}")
        
        intrinsics = np.array([
            raw_intrinsics.fx, 0, raw_intrinsics.ppx,
            0, raw_intrinsics.fy, raw_intrinsics.ppy,
            0, 0, 1
        ]).reshape(3, 3)
        
        return intrinsics

    def stop(self):
        """
        安全地停止相机和后台线程
        """
        if not self.running:
            return

        print("[D435] 正在停止相机...")
        self.running = False
        
        if self.thread is not None:
            self.thread.join()  # 等待线程结束
            
        if self.pipeline is not None:
            self.pipeline.stop()
            
        print("[D435] ✅ 相机已安全停止。")

    def __del__(self):
        """
        析构函数：确保相机资源被释放
        """
        self.stop()


if __name__ == '__main__':
    """
    测试代码：运行此文件可以测试 D435 相机
    """
    print("=" * 50)
    print("RealSense D435 相机测试")
    print("=" * 50)
    
    # 创建相机对象（可以修改分辨率和帧率）
    mycamera = Camera(width=640, height=480, fps=30)
    
    print("\n开始可视化（按 ESC 退出）...")
    try:
        while True:
            mycamera.plot_image()
            
            # 按 ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        mycamera.stop()
        cv2.destroyAllWindows()
        print("测试完成！")
